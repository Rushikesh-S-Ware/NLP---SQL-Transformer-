"""
Central configuration for NLP-SQL Transformer.
All tuneable constants live here — change them in one place.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]

# Where to look for the model.  Priority:
#   1. HF_MODEL_ID env var  (e.g. "Rushikesh-S-Ware/NLP-SQL-Transformer")
#   2. LOCAL_CHECKPOINT_DIR (extracted zip)
#   3. Bundled zip in repo  (Model_Checkpoint.zip.zip)
HF_MODEL_ID: str | None = os.getenv("HF_MODEL_ID")          # set in HF Space secrets
LOCAL_CHECKPOINT_DIR = ROOT_DIR / "checkpoint"               # extracted locally
BUNDLED_ZIP = ROOT_DIR / "Model_Checkpoint.zip.zip"          # double-zipped in repo

# ── Data limits ───────────────────────────────────────────────────────────────
MAX_ROWS_PER_TABLE = 50_000   # rows read from a single CSV
CSV_CHUNK_SIZE     = 10_000   # read CSVs in chunks of this many rows
MAX_QUERY_RESULTS  = 1_000    # max rows returned from any SQL query
MAX_SCHEMA_SAMPLES = 3        # sample values per column shown in prompt

# ── Model generation ──────────────────────────────────────────────────────────
MAX_INPUT_LENGTH   = 512
MAX_OUTPUT_LENGTH  = 256
NUM_BEAMS          = 5
GENERATION_TIMEOUT = 30       # seconds before we give up and raise TimeoutError

# ── SQL safety ────────────────────────────────────────────────────────────────
# Any query starting with one of these verbs is blocked
BLOCKED_STATEMENTS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "CREATE", "REPLACE", "TRUNCATE", "ATTACH", "DETACH",
}
