"""
Central configuration for the Spider 1.0 Text-to-SQL pipeline.

All path and model settings live here; override via environment variables
or by editing this file before running.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv optional; use exported env vars instead

# ── Repo roots ─────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).parent
PROJECT_ROOT   = REPO_ROOT.parent                   # /Users/Brian/Desktop/Text2SQL

# ── Spider 1.0 dataset paths ──────────────────────────────────────────────────
SPIDER_DATA_DIR    = PROJECT_ROOT / "spider1" / "dataset-spider1" / "spider_data"
TABLES_JSON        = SPIDER_DATA_DIR / "tables.json"
TRAIN_SPIDER_JSON  = SPIDER_DATA_DIR / "train_spider.json"
TRAIN_OTHERS_JSON  = SPIDER_DATA_DIR / "train_others.json"
DEV_JSON           = SPIDER_DATA_DIR / "dev.json"
TEST_JSON          = SPIDER_DATA_DIR / "test.json"
DATABASE_DIR       = SPIDER_DATA_DIR / "database"
TEST_DATABASE_DIR  = SPIDER_DATA_DIR / "test_database"
DEV_GOLD_SQL       = SPIDER_DATA_DIR / "dev_gold.sql"
TEST_GOLD_SQL      = SPIDER_DATA_DIR / "test_gold.sql"

# ── Embedding model (snowflake-arctic-embed-m) ───────────────────────────────
ARCTIC_MODEL_DIR  = REPO_ROOT / "models" / "arctic-embed-m"   # downloaded weights
ARCTIC_HF_ID      = "Snowflake/snowflake-arctic-embed-m"
EMBED_DEVICE      = os.getenv("EMBED_DEVICE", "mps")
EMBED_INT8        = os.getenv("EMBED_INT8", "false").lower() == "true"

# ── Vector index ──────────────────────────────────────────────────────────────
VEC_INDEX_DIR = REPO_ROOT / "vec_index"    # one .db file per Spider database

# ── Schema retrieval settings ─────────────────────────────────────────────────
SCHEMA_TOP_K       = int(os.getenv("SCHEMA_TOP_K", "5"))         # tables to retrieve (stage 1)
USE_SAMPLE_VALUES  = os.getenv("USE_SAMPLE_VALUES", "true").lower() == "true"   # inject DB samples at index time
USE_COLUMN_PRUNING = os.getenv("USE_COLUMN_PRUNING", "true").lower() == "true"  # LLM stage-2 column pruning

# ── minilm_core (legacy, kept for reference) ──────────────────────────────────
MINILM_CORE_DIR  = PROJECT_ROOT / "minilm_core"
MINILM_MODEL_DIR = MINILM_CORE_DIR / "models" / "minilm"
EMBED_PROFILE    = os.getenv("EMBED_PROFILE", "MP-Balanced")   # unused when arctic is active

# ── Few-shot settings ─────────────────────────────────────────────────────────
FEW_SHOT_K          = int(os.getenv("FEW_SHOT_K", "5"))          # examples per prompt
FEW_SHOT_POOL_FACTOR = int(os.getenv("FEW_SHOT_POOL_FACTOR", "20"))  # candidate pool = k*factor
MASK_QUESTION       = os.getenv("MASK_QUESTION", "true").lower() == "true"

# ── LLM settings ─────────────────────────────────────────────────────────────
LLM_BACKEND  = os.getenv("LLM_BACKEND", "ollama_native")  # openai | ollama | ollama_native | llama_cpp
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen3.5")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ── llama.cpp / GGUF settings (used when LLM_BACKEND=llama_cpp) ───────────────
GGUF_REPO_ID     = os.getenv("GGUF_REPO_ID",  "byteshape/Qwen3.5-9B-GGUF")
GGUF_FILENAME    = os.getenv("GGUF_FILENAME", "Qwen3.5-9B-IQ3_S-2.81bpw.gguf")
GGUF_LOCAL_DIR   = REPO_ROOT / "models" / "gguf"    # cached GGUF weights
GGUF_N_GPU_LAYERS = int(os.getenv("GGUF_N_GPU_LAYERS", "-1"))  # -1 = all layers on Metal
GGUF_N_CTX        = int(os.getenv("GGUF_N_CTX", "4096"))

# ── Self-consistency settings ─────────────────────────────────────────────────
SELF_CONSISTENCY_N    = int(os.getenv("SELF_CONSISTENCY_N", "1"))  # >1 enables voting
SELF_CONSISTENCY_TEMP = float(os.getenv("SELF_CONSISTENCY_TEMP", "0.7"))

# ── 2-pass DAIL selector ───────────────────────────────────────────────────────
TWO_PASS_SELECTOR  = os.getenv("TWO_PASS_SELECTOR", "false").lower() == "true"
SKELETON_THRESHOLD = float(os.getenv("SKELETON_THRESHOLD", "0.85"))

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR      = REPO_ROOT / "output"
PREDICTIONS_FILE = OUTPUT_DIR / "predicted_sql.txt"
RESULTS_FILE     = OUTPUT_DIR / "results.json"
