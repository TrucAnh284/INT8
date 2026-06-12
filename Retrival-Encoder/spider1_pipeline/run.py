"""
Spider 1.0 Text-to-SQL — CLI entry point.

Commands:
  index         Build MiniLM sqlite-vec indexes for all Spider 1.0 databases
  download-model  Download all-MiniLM-L6-v2 weights to models/minilm/
  run           Run inference on dev/test split
  evaluate      Evaluate existing predictions file
  demo          Interactive single-question demo

Usage examples:
  python run.py download-model
  python run.py index
  python run.py index --db_id concert_singer --force
  python run.py run --split dev --few_shot --k 5
  python run.py run --split dev --no_retriever --k 0
  python run.py evaluate --split dev
  python run.py demo
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make spider1_pipeline the working package root
sys.path.insert(0, str(Path(__file__).parent))


# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_download_model(args):
    """Download snowflake-arctic-embed-m to models/arctic-embed-m/."""
    from config import ARCTIC_MODEL_DIR, ARCTIC_HF_ID
    print(f"[download-model] Downloading {ARCTIC_HF_ID} → {ARCTIC_MODEL_DIR}")
    ARCTIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ARCTIC_HF_ID)
        model.save(str(ARCTIC_MODEL_DIR))
        print(f"[download-model] Saved to {ARCTIC_MODEL_DIR}")
    except Exception as e:
        print(f"[download-model] ERROR: {e}")
        sys.exit(1)


def cmd_download_gguf(args):
    """Download Qwen3.5-9B GGUF model for llama-cpp-python (Metal GPU)."""
    from config import GGUF_REPO_ID, GGUF_FILENAME, GGUF_LOCAL_DIR
    repo_id  = args.repo_id  or GGUF_REPO_ID
    filename = args.filename or GGUF_FILENAME
    out_dir  = GGUF_LOCAL_DIR
    out_path = out_dir / filename

    if out_path.exists() and not args.force:
        print(f"[download-gguf] Already exists: {out_path}")
        print(f"[download-gguf] Use --force to re-download.")
        return

    print(f"[download-gguf] {repo_id}/{filename} → {out_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(out_dir),
        )
        print(f"[download-gguf] Saved to {path}")
        print(f"[download-gguf] To use: set LLM_BACKEND=llama_cpp in .env")
    except Exception as e:
        print(f"[download-gguf] ERROR: {e}")
        import sys; sys.exit(1)


def cmd_index(args):
    """Build arctic-embed-m BLOB indexes for Spider 1.0 schemas."""
    from config import (
        TABLES_JSON, ARCTIC_MODEL_DIR, VEC_INDEX_DIR,
        EMBED_DEVICE, DATABASE_DIR, USE_SAMPLE_VALUES,
    )
    from embed.spider1_indexer import build_spider1_indexes

    if not ARCTIC_MODEL_DIR.exists():
        print(f"[index] arctic-embed-m not found at {ARCTIC_MODEL_DIR}. "
              f"Run: python3 run.py download-model")
        sys.exit(1)

    database_dir = DATABASE_DIR if (USE_SAMPLE_VALUES and not args.no_samples) else None

    build_spider1_indexes(
        tables_json=TABLES_JSON,
        model_dir=ARCTIC_MODEL_DIR,
        index_dir=VEC_INDEX_DIR,
        force=args.force,
        db_id_filter=args.db_id,
        device=args.device or EMBED_DEVICE,
        database_dir=database_dir,
    )


def cmd_run(args):
    """Run full Text-to-SQL inference pipeline."""
    # ── Override env vars FIRST — config.py reads them at import time ──────────
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["LLM_BASE_URL"] = args.base_url
    if args.k is not None:
        os.environ["FEW_SHOT_K"] = str(args.k)
    if args.sc_n is not None:
        os.environ["SELF_CONSISTENCY_N"] = str(args.sc_n)
    if args.top_k is not None:
        os.environ["SCHEMA_TOP_K"] = str(args.top_k)
    if getattr(args, "no_pruning", False):
        os.environ["USE_COLUMN_PRUNING"] = "false"
    if getattr(args, "two_pass", False):
        os.environ["TWO_PASS_SELECTOR"] = "true"

    # ── Now safe to import config (env vars already set) ───────────────────────
    from config import (
        DEV_JSON, TRAIN_SPIDER_JSON,
        OUTPUT_DIR, PREDICTIONS_FILE,
        LLM_BACKEND, LLM_MODEL,
    )

    from pipeline import Text2SQLPipeline

    pipe = Text2SQLPipeline.from_config(
        few_shot=(args.k != 0 if args.k is not None else True),
        use_retriever=not args.no_retriever,
        skip_selector_encoding=False,
        two_pass=getattr(args, "two_pass", False),
    )

    split_map = {"dev": DEV_JSON, "train": TRAIN_SPIDER_JSON}
    data_json = Path(args.input) if args.input else split_map.get(args.split, DEV_JSON)
    out_file  = Path(args.output) if args.output else PREDICTIONS_FILE

    pred_path = pipe.run(
        split=args.split,
        data_json=data_json,
        output_file=out_file,
        verbose=not args.quiet,
        max_samples=args.max_samples,
    )

    if args.evaluate:
        pipe.evaluate(split=args.split, pred_file=pred_path)


def cmd_evaluate(args):
    """Evaluate a predictions file against the Spider dev set."""
    from config import DEV_JSON, DATABASE_DIR, PREDICTIONS_FILE, RESULTS_FILE
    from evaluation.evaluator import evaluate_predictions, save_report

    data_json = Path(args.input) if args.input else DEV_JSON
    pred_file = Path(args.pred) if args.pred else PREDICTIONS_FILE
    db_dir    = Path(args.db_dir) if args.db_dir else DATABASE_DIR

    report = evaluate_predictions(
        dev_json=data_json,
        pred_sql_file=pred_file,
        database_dir=db_dir,
        verbose=args.verbose,
    )
    print("\n" + "=" * 50)
    print(report.summary())
    print("=" * 50)

    out = Path(args.output) if args.output else RESULTS_FILE
    save_report(report, out)
    print(f"\nDetailed results saved to: {out}")


def cmd_demo(args):
    """Interactive single-question demo."""
    from config import DEV_JSON, DATABASE_DIR
    import json

    from pipeline import Text2SQLPipeline

    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    pipe = Text2SQLPipeline.from_config(
        few_shot=True,
        use_retriever=True,
        skip_selector_encoding=False,
    )

    print("\n" + "=" * 60)
    print("Spider 1.0 Text-to-SQL Demo  (type 'quit' to exit)")
    print("=" * 60)

    # List available databases
    db_ids = sorted(pipe.schemas.keys())
    print(f"\nAvailable databases ({len(db_ids)}):")
    for i, db in enumerate(db_ids[:20]):
        print(f"  {db}", end="\n" if (i + 1) % 4 == 0 else "  ")
    if len(db_ids) > 20:
        print(f"\n  ... and {len(db_ids) - 20} more")

    while True:
        print()
        db_id = input("Database (e.g. concert_singer): ").strip()
        if db_id.lower() in ("quit", "exit", "q"):
            break
        if db_id not in pipe.schemas:
            print(f"  Unknown db_id. Choose from: {', '.join(db_ids[:10])} ...")
            continue

        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break

        print("\n[Generating SQL ...]")
        sql = pipe.predict_one(question, db_id)
        print(f"\nSQL: {sql}")

        # Optionally execute on local DB
        db_path = DATABASE_DIR / db_id / f"{db_id}.sqlite"
        if db_path.exists():
            import sqlite3
            try:
                conn = sqlite3.connect(str(db_path))
                conn.text_factory = lambda b: b.decode(errors="ignore")
                cur = conn.cursor()
                cur.execute(sql)
                rows = cur.fetchmany(10)
                conn.close()
                print(f"Result ({len(rows)} rows):")
                for row in rows:
                    print(f"  {row}")
            except Exception as e:
                print(f"Execution error: {e}")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spider 1.0 Text-to-SQL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download-model
    p_dl = sub.add_parser("download-model", help="Download snowflake-arctic-embed-m weights")

    # download-gguf
    p_dg = sub.add_parser("download-gguf", help="Download Qwen3.5-9B GGUF for llama-cpp (Metal GPU)")
    p_dg.add_argument("--repo_id",  default=None, help="HuggingFace repo (overrides config)")
    p_dg.add_argument("--filename", default=None, help="GGUF filename (overrides config)")
    p_dg.add_argument("--force",    action="store_true", help="Re-download if already cached")

    # index
    p_idx = sub.add_parser("index", help="Build sqlite-vec schema indexes")
    p_idx.add_argument("--db_id",      default=None, help="Index a single database only")
    p_idx.add_argument("--force",      action="store_true", help="Rebuild existing indexes")
    p_idx.add_argument("--device",     default=None, help="Device: cpu | mps | cuda")
    p_idx.add_argument("--no_samples", action="store_true",
                       help="Skip sample-value metadata injection (faster, less accurate)")

    # run
    p_run = sub.add_parser("run", help="Run inference on a Spider split")
    p_run.add_argument("--split",       default="dev", choices=["dev", "train"])
    p_run.add_argument("--input",       default=None,  help="Override input JSON path")
    p_run.add_argument("--output",      default=None,  help="Override output predictions path")
    p_run.add_argument("--backend",     default=None,  help="LLM backend: openai | ollama")
    p_run.add_argument("--model",       default=None,  help="LLM model name")
    p_run.add_argument("--api_key",     default=None,  help="OpenAI API key")
    p_run.add_argument("--base_url",    default=None,  help="LLM API base URL")
    p_run.add_argument("--k",           type=int, default=None, help="Few-shot K (0=zero-shot)")
    p_run.add_argument("--sc_n",        type=int, default=None, help="Self-consistency N candidates")
    p_run.add_argument("--top_k",       type=int, default=None, help="Schema top-K tables")
    p_run.add_argument("--no_retriever", action="store_true",   help="Disable schema linking")
    p_run.add_argument("--evaluate",    action="store_true",    help="Run evaluation after inference")
    p_run.add_argument("--quiet",       action="store_true",    help="Suppress progress output")
    p_run.add_argument("--max_samples",  type=int, default=None, help="Limit number of questions")
    p_run.add_argument("--no_pruning",   action="store_true",    help="Disable LLM column pruning (faster, 1 LLM call/question)")
    p_run.add_argument("--two-pass",     dest="two_pass", action="store_true",
                       help="2-pass DAIL selector: preliminary SQL → skeleton filter (threshold=0.85)")

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Evaluate a predictions file")
    p_ev.add_argument("--pred",    default=None, help="Predictions file path")
    p_ev.add_argument("--input",   default=None, help="Gold data JSON path")
    p_ev.add_argument("--db_dir",  default=None, help="Database directory path")
    p_ev.add_argument("--output",  default=None, help="Results JSON output path")
    p_ev.add_argument("--verbose", action="store_true")

    # demo
    p_dm = sub.add_parser("demo", help="Interactive single-question demo")
    p_dm.add_argument("--backend",  default=None)
    p_dm.add_argument("--model",    default=None)
    p_dm.add_argument("--api_key",  default=None)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "download-model": cmd_download_model,
        "download-gguf":  cmd_download_gguf,
        "index":          cmd_index,
        "run":            cmd_run,
        "evaluate":       cmd_evaluate,
        "demo":           cmd_demo,
    }
    dispatch[args.command](args)
