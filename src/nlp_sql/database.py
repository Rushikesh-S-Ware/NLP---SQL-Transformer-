"""
CSV -> SQLite ingestion, schema extraction, and safe query execution.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import IO

import pandas as pd

from .config import (
    CSV_CHUNK_SIZE,
    MAX_QUERY_RESULTS,
    MAX_ROWS_PER_TABLE,
    MAX_SCHEMA_SAMPLES,
    BLOCKED_STATEMENTS,
)

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "int64":   "INTEGER",
    "float64": "REAL",
    "bool":    "INTEGER",
    "object":  "TEXT",
    "datetime64[ns]": "TEXT",
}


def _friendly_dtype(dtype) -> str:
    return _DTYPE_MAP.get(str(dtype), "TEXT")


def load_csv_to_table(
    source,
    conn: sqlite3.Connection,
    table_name: str,
    max_rows: int = MAX_ROWS_PER_TABLE,
    chunk_size: int = CSV_CHUNK_SIZE,
) -> pd.DataFrame:
    """Read a CSV (in chunks) into SQLite table. Raises ValueError on bad input."""
    chunks = []
    rows_seen = 0

    try:
        reader = pd.read_csv(source, chunksize=chunk_size, low_memory=False)
        for chunk in reader:
            if rows_seen >= max_rows:
                logger.warning("Table '%s': hit row cap (%d). Extra rows skipped.", table_name, max_rows)
                break
            remaining = max_rows - rows_seen
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            chunks.append(chunk)
            rows_seen += len(chunk)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV '{table_name}' is empty or has no parseable data.") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read CSV '{table_name}': {exc}") from exc

    # pandas may yield one empty chunk for headers-only CSVs
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    if df.empty:
        raise ValueError(f"CSV '{table_name}' produced zero rows.")

    df.to_sql(table_name, conn, index=False, if_exists="replace")
    logger.info("Loaded %d rows into table '%s'.", len(df), table_name)
    return df


def load_files_to_sqlite(
    files: list,
    max_rows: int = MAX_ROWS_PER_TABLE,
) -> tuple:
    """Load each file as table t1, t2, ... into a fresh in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    tables = {}
    for idx, f in enumerate(files):
        table_name = f"t{idx + 1}"
        df = load_csv_to_table(f, conn, table_name, max_rows=max_rows)
        tables[table_name] = df
    return conn, tables


def build_schema_string(
    tables: dict,
    n_samples: int = MAX_SCHEMA_SAMPLES,
) -> str:
    """Build a compact schema string: t1(col:TYPE[sample1,sample2,...]) | t2(...)"""
    parts = []
    for tbl, df in tables.items():
        col_descs = []
        for col in df.columns:
            dtype_tag = _friendly_dtype(df[col].dtype)
            samples = df[col].dropna().astype(str).unique()[:n_samples].tolist()
            sample_str = ",".join(samples) if samples else ""
            col_descs.append(f"{col}:{dtype_tag}[{sample_str}]")
        parts.append(f"{tbl}({', '.join(col_descs)})")
    return " | ".join(parts)


def _inject_limit(sql: str, limit: int = MAX_QUERY_RESULTS) -> str:
    """Append LIMIT to a SELECT if it does not already have one."""
    stripped = sql.strip().rstrip(";")
    if re.search(r"\bLIMIT\b", stripped, re.IGNORECASE):
        return sql
    if re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        return f"{stripped} LIMIT {limit}"
    return sql


def validate_query(sql: str) -> None:
    """Raise ValueError if sql is empty or is a write/DDL statement."""
    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")
    first_word = sql.strip().split()[0].upper()
    if first_word in BLOCKED_STATEMENTS:
        raise ValueError(
            f"Query blocked: '{first_word}' statements are not permitted. "
            "Only SELECT queries are allowed."
        )


def execute_query(
    sql: str,
    conn: sqlite3.Connection,
    limit: int = MAX_QUERY_RESULTS,
) -> pd.DataFrame:
    """Validate, inject LIMIT, execute SQL and return a DataFrame."""
    if not sql or not sql.strip():
        raise ValueError("SQL query is empty.")
    validate_query(sql)
    safe_sql = _inject_limit(sql, limit)
    try:
        df = pd.read_sql_query(safe_sql, conn)
    except Exception as exc:
        raise RuntimeError(f"Query execution failed: {exc}\nSQL: {safe_sql}") from exc
    return df
