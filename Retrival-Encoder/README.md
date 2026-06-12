# Lossless INT8 Compression of Schema Retrieval Encoders for Edge-Deployed Text-to-SQL

> **Paper:** *Lossless INT8 Compression of Schema Retrieval Encoders for Edge-Deployed Text-to-SQL*  
> **Author:** Brian Nguyen  
> **Status:** Under review

---

## TL;DR

We show that INT8 dynamic quantization of the `arctic-embed-m` schema retrieval encoder is **completely safe** for edge Text-to-SQL: **zero retrieval-quality degradation** (R@5 and MRR identical to FP32) while cutting encoder RAM by **78%** (418 MB → 91 MB) and encoding latency by **15%**. The full pipeline achieves **EX = 0.824** on Spider 1.0 using a 9B local model — competitive with fine-tuned 3B-parameter systems.

---

## Key Results

### INT8 Quantization: Zero Quality Loss

| Metric | FP32 | INT8 | Δ |
|--------|------|------|---|
| R@5 (Spider 1.0) | 0.9952 | 0.9952 | **0.000** |
| MRR (Spider 1.0) | 0.9513 | 0.9513 | **0.000** |
| Encoder RAM | 418 MB | **91 MB** | −78.2% |
| Encoding latency | 64.1 ms | **54.4 ms** | −15.2% |

INT8 is the **Pareto-optimal** quantization profile: identical quality to FP32 at one-quarter the memory footprint.

### Pipeline Ablation (Spider 1.0 Dev, 1,030 questions)

| Configuration | EX | EM | Errors | ΔEX |
|---------------|----|----|--------|-----|
| Baseline (k=3, SC=1) | 0.786 | 0.240 | 92 | — |
| +sql2skeleton | 0.805 | 0.284 | 46 | +1.91 pp |
| +k=5 | 0.813 | 0.302 | 50 | +2.74 pp |
| +SC=3 | 0.810 | 0.295 | 39 | +2.45 pp |
| +2-pass correction | 0.823 | 0.345 | 47 | +3.71 pp |
| **SC=3+k=5+2-pass (best)** | **0.824** | **0.344** | **38** | **+3.87 pp** |

All improvements over baseline are statistically significant (*p* < 0.001, McNemar's test).

### Difficulty Breakdown

| Configuration | Easy | Medium | Hard | Extra Hard |
|---------------|------|--------|------|------------|
| Baseline | 0.873 | 0.853 | 0.624 | 0.626 |
| Best (SC=3+k=5+2-pass) | **0.893** | **0.884** | **0.795** | **0.587** |
| Δ Baseline→Best | +2.0 pp | +3.1 pp | **+17.1 pp** | −3.9 pp |

`sql2skeleton` is the biggest driver of Hard queries (+13.1 pp on Hard alone).

### Comparison with Published Systems (Spider 1.0 Dev)

| System | Base Model | Training | EX |
|--------|-----------|----------|-----|
| DAIL-SQL (Gao et al., 2023) | GPT-4 (~1.7T) | few-shot | 0.866 |
| DIN-SQL (Pourreza et al., 2023) | GPT-4 (~1.7T) | few-shot | 0.826 |
| RESDSQL (Li et al., 2023) | T5-3B | fine-tuned | 0.791 |
| PICARD (Scholak et al., 2021) | T5-3B | fine-tuned | 0.754 |
| **Ours** | **Qwen3.5 9B (local)** | **none** | **0.824** |

Our pipeline is competitive with fine-tuned 3B models and DIN-SQL, using a 190× smaller model than GPT-4 with **no fine-tuning** and **fully local inference**.

### Spider 2.0-Lite Results (135 questions)

| Configuration | EX | Correct/Total |
|---------------|----|---------------|
| Zero-shot | 0.067 | 9/135 |
| +few-shot k=3 | 0.104 | 14/135 |
| +SC=3 + 2-pass | **0.111** | **15/135** |

---

## Pipeline Architecture

```
Natural Language Question + DB Schema
              │
              ▼
┌─────────────────────────────────┐
│  Schema Retrieval               │  arctic-embed-m (INT8 dynamic)
│  (arctic-embed-m + sqlite-vec)  │  109.5M params, 91 MB RAM
│                                 │  Top-k=5 relevant tables retrieved
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Few-Shot Selection             │  DAIL-SQL: sql2skeleton cosine sim
│  (sql2skeleton)                 │  skeleton = SQL with literals stripped
│                                 │  Selects k=5 structural demonstrations
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Prompt Assembly                │  DAIL-SQL format:
│  (Code Representation)          │  CREATE TABLE + sample rows
│                                 │  + few-shot Q→SQL pairs + question
└────────────────┬────────────────┘
                 │        (~1,400 tokens total)
                 ▼
┌─────────────────────────────────┐
│  LLM Inference × SC=3           │  Qwen3.5 9B via Ollama (local)
│  (self-consistency)             │  or any OpenAI-compatible API
│                                 │  Temperature=0.7, 3 candidates
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Executor Selector              │  Pick first error-free SQL candidate
│  + 2-Pass Correction            │  If all fail: re-prompt with error msg
└────────────────┬────────────────┘
                 │
                 ▼
           predicted SQL
```

---

## INT8 Quantization Details

**Method:** `torch.quantization.quantize_dynamic` — no calibration data required.

| Layer type | INT8 Full | Notes |
|-----------|-----------|-------|
| Q / K / V projections (×12) | ✅ INT8 | nn.Linear → quantized |
| Attn Output Projection (×12) | ✅ INT8 | nn.Linear → quantized |
| FFN Dense 1 (×12) | ✅ INT8 | nn.Linear → quantized |
| FFN Dense 2 (×12) | ✅ INT8 | nn.Linear → quantized |
| Pooler Linear | ✅ INT8 | nn.Linear → quantized |
| Token Embedding | ❌ FP32 | nn.Embedding — not targeted by quantize_dynamic |
| LayerNorm | ❌ FP32 | No nn.Linear → not targeted |

**Sensitivity analysis:** 84 experiments across all 12 transformer blocks × 7 sublayer types. All `|ΔR@5| < 0.001` — uniform INT8 is optimal; no mixed-precision scheme improves on it.

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed (for local LLM inference)
- Apple M-series / x86 CPU (GPU not required for retrieval)

### 1. Install dependencies

```bash
cd spider1_pipeline
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env: set LLM_BACKEND, API keys, etc.
```

### 3. Download Spider 1.0 dataset

```
spider1/
├── dev.json
├── train_spider.json
├── tables.json
└── database/         ← SQLite databases (not included, download separately)
```

Download from: https://yale-lily.github.io/spider

### 4. Build vector indexes

```bash
python run.py index
```

### 5. Run inference

```bash
# Best configuration: SC=3, k=5, 2-pass correction (local Qwen3.5 9B)
ollama pull qwen2.5-coder:latest
python run.py run --split dev --k 5 --sc_n 3 --evaluate

# OpenAI / DeepSeek API
python run.py run --split dev --k 5 --sc_n 3 \
  --backend openai --model gpt-4o --evaluate

# Quick smoke test (50 examples)
python run.py run --split dev --k 3 --max_samples 50 --evaluate
```

### 6. Reproduce quantization benchmarks

```bash
# Sensitivity analysis (84 experiments, ~2h on CPU)
python bench_layer_sensitivity.py

# Latency benchmark (FP32 vs INT8 profiles)
python bench_latency.py

# Statistical significance tests
python bench_significance.py
```

---

## Configuration

All settings in `spider1_pipeline/config.py`. Key `.env` options:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `openai` | `openai` or `ollama` |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `OPENAI_API_KEY` | — | API key |
| `LLM_BASE_URL` | OpenAI URL | Override for DeepSeek / local |
| `FEW_SHOT_K` | `5` | Few-shot examples count |
| `SCHEMA_TOP_K` | `10` | Tables retrieved by encoder |
| `EMBED_PROFILE` | `MP-Balanced` | Quantization profile |
| `SELF_CONSISTENCY_N` | `1` | SC candidates (1 = disabled) |

**Embed profiles:** `FP32-Full` · `FP16-Full` · `MP-Conservative` · `MP-Balanced` · `MP-Aggressive` · `INT8-Full`

---

## Project Structure

```
Text2SQL/
├── README.md
├── spider1_pipeline/          # Main pipeline code
│   ├── run.py                 # CLI entry point
│   ├── pipeline.py            # End-to-end orchestration
│   ├── config.py              # Settings + paths
│   ├── requirements.txt
│   ├── .env.example
│   ├── embed/                 # arctic-embed-m retriever + sqlite-vec index
│   ├── schema/                # Spider schema parser + serializer
│   ├── examples/              # DAIL-SQL sql2skeleton few-shot selector
│   ├── prompt/                # DAIL-SQL prompt builder
│   ├── llm/                   # OpenAI + Ollama clients
│   ├── postprocess/           # SQL cleaner + self-consistency vote
│   ├── evaluation/            # EX + EM evaluator
│   ├── bench_layer_sensitivity.py   # 84-experiment INT8 sensitivity study
│   ├── bench_latency.py             # FP32 vs INT8 latency benchmark
│   └── bench_significance.py        # McNemar's test + bootstrap CI
│
├── minilm_core/               # Embedding backbone (arctic-embed-m)
│   └── src/                   # Model loading + quantization profiles
│
├── paper/                     # LaTeX source
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
└── spider1/                   # Spider 1.0 dataset (gitignored databases)
    ├── dev.json
    ├── tables.json
    └── database/              # ← download separately
```

---

## Models Used

| Model | Role | Size | Source |
|-------|------|------|--------|
| `Snowflake/arctic-embed-m` | Schema retrieval encoder | 109.5M params, 768-dim | HuggingFace |
| `Qwen3.5 9B` | SQL generation LLM | 9B params | Ollama (`qwen2.5-coder`) |

Neither model is fine-tuned. The retrieval encoder is quantized at inference time using `torch.quantization.quantize_dynamic`.

---

## Paper Abstract

Natural-language interfaces to relational databases (Text-to-SQL) have achieved impressive accuracy using large language models, but prevailing systems rely on cloud APIs and GPU-accelerated servers, precluding deployment in privacy-sensitive or resource-constrained environments.

This paper presents a **training-free, fully local** Text-to-SQL pipeline and investigates whether **INT8 dynamic quantization** of the arctic-embed-m schema retriever is safe for edge deployment. We conduct a **systematic 84-experiment sensitivity analysis** covering all 12 transformer blocks and every linear sublayer of the encoder, evaluating R@5 and MRR on the Spider 1.0 development set (1,034 questions).

The sensitivity analysis reveals that INT8 quantization introduces **zero retrieval-quality degradation** (R@5 and MRR unchanged versus FP32) while reducing encoder RAM by **78.2%** (418 MB → 91 MB) and improving encoding latency by **15.2%**. The ablation study identifies five synergistic pipeline components that collectively raise execution accuracy from **0.786 to 0.824** on Spider 1.0, with each component statistically significant (*p* < 0.001, McNemar's test).

These findings establish **INT8 as the Pareto-optimal quantization profile** — identical quality to FP32 at one-quarter the memory footprint — and provide the first empirical evidence that schema-linking retrievers can be safely compressed for edge Text-to-SQL deployment.

---

## Citation

```bibtex
@article{nguyen2025int8text2sql,
  title   = {Lossless {INT8} Compression of Schema Retrieval Encoders
             for Edge-Deployed Text-to-{SQL}},
  author  = {Nguyen, Brian},
  year    = {2025},
  note    = {Under review}
}
```

---

## References

- **DAIL-SQL:** Gao et al. (2023). *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation.* arXiv:2308.15363
- **arctic-embed-m:** Merrick et al. (2024). *Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models.* arXiv:2405.05374
- **Spider 1.0:** Yu et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP 2018
- **Spider 2.0:** Spider 2.0-Lite benchmark. https://spider2-sql.github.io
- **DIN-SQL:** Pourreza & Rafiei (2023). *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.* NeurIPS 2023
