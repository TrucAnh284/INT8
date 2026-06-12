# Spider 1.0 Text-to-SQL Pipeline

A production-grade Text-to-SQL pipeline for the **Spider 1.0** benchmark, implementing the **DAIL-SQL** methodology with `minilm_core` as the embedding backbone.

---

## Architecture

```
Question + DB Schema
        │
        ▼
┌──────────────────────┐
│  Schema Linking      │  all-MiniLM-L6-v2 (mixed-precision)
│  (minilm_core embed) │  → Top-K relevant tables retrieved
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Few-Shot Selector   │  DAIL selection:
│  (examples/)         │  masked-question cosine sim
└──────────┬───────────┘  + optional SQL skeleton filter
           │
           ▼
┌──────────────────────┐
│  Prompt Builder      │  Code Representation Prompt
│  (prompt/)           │  CREATE TABLE + Q/SQL pairs
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM Inference       │  OpenAI (GPT-4o / DeepSeek)
│  (llm/)              │  or Ollama local models
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Post-process        │  SQL extract + clean
│  (postprocess/)      │  + self-consistency vote
└──────────┬───────────┘
           │
           ▼
      predicted SQL
```

## Research Basis

| Paper | Method | Dev EX | Test EX |
|-------|--------|--------|---------|
| DAIL-SQL (Gao et al., 2023) | GPT-4 + DAIL selection | 83.1% | 86.2% |
| DAIL-SQL + SC (Gao et al., 2023) | GPT-4 + self-consistency | 83.6% | **86.6%** |
| C3SQL (Dong et al., 2023) | ChatGPT + zero-shot + SC | — | ~82% |
| DIN-SQL (Pourreza et al., 2023) | GPT-4 + decomposition | — | ~82% |
| RESDSQL (Li et al., 2023) | Fine-tuned T5 | — | ~79% |

Papers in `../spider1/research/`:
- `2308.15363v4.pdf` — **DAIL-SQL** (our primary reference)
- `2307.07306v1.pdf` — **C3SQL** (zero-shot + self-consistency)
- `2304.11015v3.pdf` — **DIN-SQL** (decomposed prompting)
- `1809.08887v5.pdf` — **Spider** original dataset paper

---

## Recommended LLMs

### Via API (best accuracy)
| Model | Expected Spider EX | Cost |
|-------|--------------------|------|
| `gpt-4o` | ~85–86% | High |
| `deepseek-chat` (DeepSeek-V3) | ~84–85% | Low |
| `gpt-3.5-turbo` | ~74–76% | Very low |

### Local via Ollama (free, no API key)
| Model | Expected Spider EX | VRAM |
|-------|--------------------|------|
| `deepseek-coder:33b` | ~83–84% | 20GB |
| `qwen2.5-coder:32b` | ~82–83% | 20GB |
| `codestral:22b` | ~80–81% | 14GB |
| `deepseek-coder:6.7b` | ~75–76% | 6GB |

---

## Quick Start

### 1. Install dependencies
```bash
cd spider1_pipeline
pip install -r requirements.txt
```

### 2. Download MiniLM model weights
```bash
python run.py download-model
```

### 3. Build schema vector indexes
```bash
python run.py index
```

### 4. Set your API key (OpenAI) or start Ollama
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# OR: Ollama (start server + pull model)
ollama serve &
ollama pull deepseek-coder:33b
```

### 5. Run inference on Spider dev set
```bash
# Zero-shot with GPT-4o
python run.py run --split dev --k 0

# 5-shot DAIL selection with GPT-4o
python run.py run --split dev --k 5 --evaluate

# Local Ollama model
python run.py run --split dev --k 5 \
  --backend ollama --model deepseek-coder:33b --evaluate

# DeepSeek API
python run.py run --split dev --k 5 \
  --backend openai \
  --model deepseek-chat \
  --base_url https://api.deepseek.com/v1 \
  --api_key $DEEPSEEK_API_KEY \
  --evaluate
```

### 6. Evaluate predictions
```bash
python run.py evaluate --pred output/predicted_sql.txt
```

### 7. Interactive demo
```bash
python run.py demo
```

---

## Advanced Options

### Self-Consistency Voting (improves +0.5–1% EX)
```bash
python run.py run --split dev --k 5 --sc_n 5 --evaluate
```

### Schema linking: control table retrieval
```bash
# Retrieve top 10 tables (default)
python run.py run --split dev --top_k 10

# Disable schema linking (use full schema)
python run.py run --split dev --no_retriever
```

### Quick test on 50 examples
```bash
python run.py run --split dev --k 3 --max_samples 50 --evaluate
```

---

## Configuration

All defaults in `config.py`. Override any setting via environment variable:

| Env Var | Default | Description |
|---------|---------|-------------|
| `LLM_BACKEND` | `openai` | `openai` or `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `LLM_BASE_URL` | OpenAI URL | API endpoint |
| `OPENAI_API_KEY` | — | API key |
| `FEW_SHOT_K` | `5` | Few-shot examples |
| `SCHEMA_TOP_K` | `10` | Tables to retrieve |
| `EMBED_PROFILE` | `MP-Balanced` | MiniLM quantization profile |
| `SELF_CONSISTENCY_N` | `1` | SC candidates (1 = disabled) |

---

## Project Structure

```
spider1_pipeline/
├── config.py                  # All path + model settings
├── run.py                     # CLI entry point
├── pipeline.py                # End-to-end orchestration
├── requirements.txt
│
├── schema/
│   ├── loader.py              # Parse tables.json → SpiderSchema
│   └── serializer.py          # Schema → CREATE TABLE SQL string
│
├── embed/
│   ├── spider1_indexer.py     # Build sqlite-vec indexes
│   └── spider1_retriever.py   # Top-K table retrieval (wraps minilm_core)
│
├── examples/
│   └── selector.py            # DAIL few-shot selection
│
├── prompt/
│   └── builder.py             # Assemble final LLM prompt
│
├── llm/
│   ├── base.py                # Abstract client
│   ├── openai_client.py       # OpenAI / DeepSeek / compatible APIs
│   └── ollama_client.py       # Local Ollama inference
│
├── postprocess/
│   └── sql_cleaner.py         # Extract SQL, fix errors, self-consistency vote
│
├── evaluation/
│   └── evaluator.py           # EX + EM metrics
│
├── vec_index/                 # Auto-created: per-DB sqlite-vec indexes
└── output/                    # Auto-created: predictions + results
```

---

## Embedding: minilm_core

The `embed/` module wraps `../minilm_core/` for Spider 1.0:

- **Model**: `all-MiniLM-L6-v2` (384-dim, 22M params)
- **Profile**: `MP-Balanced` — FP32 embeddings + pooler, INT8 transformer layers
- **Index**: one sqlite-vec `.db` file per Spider database
- **Use case**: table-level schema linking (retrieve relevant tables given a question)
- **Latency**: ~2–5ms encode + ~1ms search per query on CPU

For the few-shot selector, the same model encodes all 8,659 training questions once at startup (takes ~30s on CPU).
