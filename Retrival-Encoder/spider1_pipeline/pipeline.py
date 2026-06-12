"""
Spider 1.0 Text-to-SQL Pipeline.

End-to-end orchestration:
  1. Load Spider schema (tables.json)
  2. Schema linking via MiniLM (top-K table retrieval)
  3. Few-shot example selection (DAIL selection)
  4. Prompt construction (Code Representation)
  5. LLM inference (with optional self-consistency)
  6. SQL post-processing
  7. Write predictions file
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from config import (
    TABLES_JSON, TRAIN_SPIDER_JSON, TRAIN_OTHERS_JSON,
    DEV_JSON, DATABASE_DIR, VEC_INDEX_DIR,
    ARCTIC_MODEL_DIR, EMBED_DEVICE, EMBED_INT8, SCHEMA_TOP_K,
    USE_SAMPLE_VALUES, USE_COLUMN_PRUNING,
    MINILM_MODEL_DIR, EMBED_PROFILE,    # legacy, used by FewShotSelector
    FEW_SHOT_K, FEW_SHOT_POOL_FACTOR, MASK_QUESTION,
    LLM_BACKEND, LLM_MODEL, LLM_BASE_URL, OPENAI_API_KEY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    GGUF_REPO_ID, GGUF_FILENAME, GGUF_LOCAL_DIR, GGUF_N_GPU_LAYERS, GGUF_N_CTX,
    SELF_CONSISTENCY_N, SELF_CONSISTENCY_TEMP,
    TWO_PASS_SELECTOR, SKELETON_THRESHOLD,
    OUTPUT_DIR, PREDICTIONS_FILE, RESULTS_FILE,
)
from schema.loader import load_tables_json, SpiderSchema
from schema.serializer import serialize_schema_code_repr
from schema.pruner import SchemaPruner
from embed.spider1_retriever import Spider1Retriever
from examples.selector import FewShotSelector, ExampleItem, sql2skeleton
from prompt.builder import PromptBuilder
from llm import get_client
from postprocess.sql_cleaner import extract_sql, fix_common_errors, self_consistency_vote
from evaluation.evaluator import evaluate_predictions, save_report


# ── Helper: build schema token sets (table + column names per DB) ──────────────

def _build_schema_tokens(schemas: dict[str, SpiderSchema]) -> dict[str, set[str]]:
    token_map: dict[str, set[str]] = {}
    for db_id, schema in schemas.items():
        tokens: set[str] = set()
        for tbl in schema.tables:
            tokens.add(tbl.name_original.lower())
            tokens.update(c.name_original.lower() for c in tbl.columns)
        token_map[db_id] = tokens
    return token_map


# ── Main pipeline ─────────────────────────────────────────────────────────────

class Text2SQLPipeline:
    """
    Spider 1.0 Text-to-SQL pipeline.

    Usage:
        pipe = Text2SQLPipeline.from_config()
        pipe.run(split="dev")
    """

    def __init__(
        self,
        schemas: dict[str, SpiderSchema],
        retriever: Optional[Spider1Retriever],
        selector: Optional[FewShotSelector],
        prompt_builder: PromptBuilder,
        llm_client,
        pruner: Optional[SchemaPruner] = None,
        schema_top_k: int = SCHEMA_TOP_K,
        few_shot_k: int = FEW_SHOT_K,
        self_consistency_n: int = SELF_CONSISTENCY_N,
        self_consistency_temp: float = SELF_CONSISTENCY_TEMP,
        two_pass: bool = False,
        database_dir: Path = DATABASE_DIR,
        output_dir: Path = OUTPUT_DIR,
    ):
        self.schemas              = schemas
        self.retriever            = retriever
        self.selector             = selector
        self.prompt_builder       = prompt_builder
        self.llm                  = llm_client
        self.pruner               = pruner
        self.schema_top_k         = schema_top_k
        self.few_shot_k           = few_shot_k
        self.self_consistency_n   = self_consistency_n
        self.self_consistency_temp = self_consistency_temp
        self.two_pass             = two_pass
        self.database_dir         = database_dir
        self.output_dir           = output_dir

    # ── Schema linking ────────────────────────────────────────────────────────

    def _get_selected_tables(self, question: str, db_id: str) -> Optional[list[str]]:
        """
        Retrieve top-K relevant table names for the question using MiniLM.
        Returns None if no retriever (use all tables).
        """
        if self.retriever is None:
            return None
        try:
            return self.retriever.retrieve_table_names(question, db_id, k=self.schema_top_k)
        except FileNotFoundError:
            return None   # index not built; fall back to full schema

    # ── Single question inference ─────────────────────────────────────────────

    def predict_one(
        self,
        question: str,
        db_id: str,
        schema_tokens: Optional[set[str]] = None,
    ) -> str:
        """
        Generate SQL for a single (question, db_id) pair.
        Returns cleaned SQL string.
        """
        schema = self.schemas.get(db_id)
        if schema is None:
            return f"-- ERROR: unknown db_id '{db_id}'"

        # 1. Stage-1: coarse table retrieval (arctic-embed-m)
        selected_tables = self._get_selected_tables(question, db_id)

        # 2. Stage-2: LLM column pruning (Qwen 3.5 identifies needed columns)
        pruned_columns: Optional[dict] = None
        if self.pruner and selected_tables:
            pruned_columns = self.pruner.prune(question, schema, selected_tables)

        # 3. Few-shot examples (pass 1: masked-question similarity, no skeleton filter)
        examples: list[ExampleItem] = []
        if self.selector and self.few_shot_k > 0:
            examples = self.selector.select(
                question=question,
                db_id=db_id,
                k=self.few_shot_k,
                schema_tokens=schema_tokens,
            )

        db_path = self.database_dir / db_id / f"{db_id}.sqlite"

        # 3b. Two-pass DAIL: generate preliminary SQL → extract skeleton → re-select examples
        if self.two_pass and self.selector:
            fp_messages = self.prompt_builder.build_messages(
                question=question,
                schema=schema,
                examples=examples,
                selected_tables=selected_tables,
                db_path=db_path if db_path.exists() else None,
                pruned_columns=pruned_columns,
            )
            fp_resp = self.llm.complete_with_retry(
                messages=fp_messages,
                temperature=0.0,
                max_tokens=LLM_MAX_TOKENS,
                n=1,
            )
            pre_skeleton = sql2skeleton(fix_common_errors(extract_sql(fp_resp.text)))
            examples = self.selector.select(
                question=question,
                db_id=db_id,
                k=self.few_shot_k,
                schema_tokens=schema_tokens,
                pre_gen_skeleton=pre_skeleton,
            )

        # 4. Build prompt with pruned schema
        messages = self.prompt_builder.build_messages(
            question=question,
            schema=schema,
            examples=examples,
            selected_tables=selected_tables,
            db_path=db_path if db_path.exists() else None,
            pruned_columns=pruned_columns,
        )

        # 5. LLM inference
        use_sc = self.self_consistency_n > 1
        temp   = self.self_consistency_temp if use_sc else LLM_TEMPERATURE
        n      = self.self_consistency_n if use_sc else 1

        response = self.llm.complete_with_retry(
            messages=messages,
            temperature=temp,
            max_tokens=LLM_MAX_TOKENS,
            n=n,
        )

        # 6. Post-process (strip <think> + extract SQL)
        if use_sc and len(response.candidates) > 1:
            candidates = [fix_common_errors(extract_sql(c)) for c in response.candidates]
            sql = self_consistency_vote(candidates, db_path) if db_path.exists() else candidates[0]
        else:
            sql = fix_common_errors(extract_sql(response.text))

        return sql

    # ── Batch inference ───────────────────────────────────────────────────────

    def run(
        self,
        split: str = "dev",
        data_json: Optional[Path] = None,
        output_file: Optional[Path] = None,
        verbose: bool = True,
        max_samples: Optional[int] = None,
    ) -> Path:
        """
        Run the full pipeline on a dataset split.

        Parameters
        ----------
        split       : "dev" | "test" | "train"
        data_json   : override path to the JSON file
        output_file : override path for prediction output
        verbose     : print progress
        max_samples : limit number of questions (for quick tests)

        Returns path to the predictions file.
        """
        split_map = {
            "dev":   DEV_JSON,
            "train": TRAIN_SPIDER_JSON,
        }
        data_path = data_json or split_map.get(split, DEV_JSON)
        out_path  = output_file or PREDICTIONS_FILE

        questions = json.loads(data_path.read_text(encoding="utf-8"))
        if max_samples:
            questions = questions[:max_samples]

        # Pre-build schema token sets for DAIL masking
        schema_tokens_map = _build_schema_tokens(self.schemas)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: skip already-written predictions (only for full runs, not quick tests)
        done_count = 0
        if out_path.exists() and max_samples is None:
            existing = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(existing) > 100 and len(existing) < len(questions):
                done_count = len(existing)
                print(f"[pipeline] Resuming from question {done_count}/{len(questions)}")

        predictions: list[str] = []
        t_start = time.perf_counter()

        fout = out_path.open("a" if done_count > 0 else "w", encoding="utf-8")
        try:
            for i, item in enumerate(questions):
                if i < done_count:
                    continue
                db_id    = item["db_id"]
                question = item["question"]
                toks     = schema_tokens_map.get(db_id, set())

                if verbose and i % 10 == 0:
                    elapsed = time.perf_counter() - t_start
                    print(f"[pipeline] {i}/{len(questions)}  ({elapsed:.1f}s)  "
                          f"db={db_id}  q={question[:50]}")

                sql = self.predict_one(question, db_id, schema_tokens=toks)
                predictions.append(sql)
                fout.write(sql + "\n")
                fout.flush()
        finally:
            fout.close()

        elapsed = time.perf_counter() - t_start
        total = done_count + len(predictions)
        print(f"[pipeline] Done: {total} predictions in {elapsed:.1f}s → {out_path}")
        return out_path

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        split: str = "dev",
        pred_file: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        """Run evaluation and print results."""
        split_map = {"dev": DEV_JSON}
        data_json = split_map.get(split, DEV_JSON)
        pred_path = pred_file or PREDICTIONS_FILE

        print(f"\n[evaluate] Evaluating {pred_path} on Spider {split} ...")
        report = evaluate_predictions(
            dev_json=data_json,
            pred_sql_file=pred_path,
            database_dir=self.database_dir,
            verbose=verbose,
        )
        print("\n" + "=" * 50)
        print(report.summary())
        print("=" * 50 + "\n")
        save_report(report, RESULTS_FILE)
        print(f"[evaluate] Full report saved to {RESULTS_FILE}")

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        few_shot: bool = True,
        use_retriever: bool = True,
        skip_selector_encoding: bool = False,
        two_pass: bool = TWO_PASS_SELECTOR,
    ) -> "Text2SQLPipeline":
        """
        Build the full pipeline from config.py settings.

        Parameters
        ----------
        few_shot               : enable few-shot example selection
        use_retriever          : enable MiniLM schema linking
        skip_selector_encoding : skip embedding the training set (for quick tests)
        """
        print("[pipeline] Loading Spider 1.0 schemas ...")
        schemas = load_tables_json(TABLES_JSON)

        # ── Stage-1: arctic-embed-m schema retriever ──────────────────────────────
        retriever: Optional[Spider1Retriever] = None
        if use_retriever:
            if not ARCTIC_MODEL_DIR.exists():
                print(f"[WARNING] arctic-embed-m not found at {ARCTIC_MODEL_DIR}. "
                      f"Schema linking disabled. Run: python3 run.py download-model")
            elif not VEC_INDEX_DIR.exists() or not any(VEC_INDEX_DIR.glob("*.db")):
                print(f"[WARNING] No vector indexes at {VEC_INDEX_DIR}. "
                      f"Schema linking disabled. Run: python3 run.py index")
            else:
                int8_tag = " INT8" if EMBED_INT8 else ""
                print(f"[pipeline] Loading arctic-embed-m{int8_tag} retriever on {EMBED_DEVICE} ...")
                retriever = Spider1Retriever(
                    model_dir=ARCTIC_MODEL_DIR,
                    index_dir=VEC_INDEX_DIR,
                    device=EMBED_DEVICE,
                    quantize_int8=EMBED_INT8,
                )

        # ── LLM client (built early — shared with pruner) ────────────────────────
        print(f"[pipeline] Connecting to LLM: {LLM_BACKEND} / {LLM_MODEL}")
        if LLM_BACKEND == "llama_cpp":
            gguf_local = GGUF_LOCAL_DIR / GGUF_FILENAME
            llm = get_client(
                backend=LLM_BACKEND,
                repo_id=GGUF_REPO_ID,
                filename=GGUF_FILENAME,
                local_path=gguf_local if gguf_local.exists() else None,
                n_gpu_layers=GGUF_N_GPU_LAYERS,
                n_ctx=GGUF_N_CTX,
            )
        else:
            llm = get_client(
                backend=LLM_BACKEND,
                model=LLM_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=LLM_BASE_URL,
            )

        # ── Stage-2: LLM column pruner ────────────────────────────────────────────
        pruner: Optional[SchemaPruner] = None
        if USE_COLUMN_PRUNING and retriever is not None:
            print("[pipeline] Column pruning enabled (stage-2 via Qwen 3.5)")
            pruner = SchemaPruner(llm_client=llm)

        # ── Few-shot selector (reuses retriever's embed model to avoid double load) ──
        selector: Optional[FewShotSelector] = None
        if few_shot and not skip_selector_encoding:
            schema_tokens_map = _build_schema_tokens(schemas)
            print("[pipeline] Building few-shot index from training data ...")
            # Reuse the already-loaded ArcticEmbedModel from retriever if available
            shared_embed = retriever._embed if retriever is not None else None
            selector = FewShotSelector.from_file(
                train_json_paths=[TRAIN_SPIDER_JSON, TRAIN_OTHERS_JSON],
                method="dail" if MASK_QUESTION else "question",
                model_dir=ARCTIC_MODEL_DIR if ARCTIC_MODEL_DIR.exists() else None,
                model=shared_embed,   # skip re-loading if we already have it
                device=EMBED_DEVICE,
                cross_domain=True,
                schema_tokens_per_db=schema_tokens_map,
                skeleton_threshold=SKELETON_THRESHOLD if two_pass else 0.0,
            )

        # ── Prompt builder (samples injected at render time if available) ───────
        prompt_builder = PromptBuilder(repr_type="code")

        if two_pass:
            print(f"[pipeline] 2-pass DAIL selector enabled (skeleton_threshold={SKELETON_THRESHOLD})")

        return cls(
            schemas=schemas,
            retriever=retriever,
            selector=selector,
            prompt_builder=prompt_builder,
            llm_client=llm,
            pruner=pruner,
            two_pass=two_pass,
        )
