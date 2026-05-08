"""
SQL post-processing: fix malformed output, then run fallback strategies.

The model sometimes produces near-correct SQL that just needs a small fix.
This module applies a chain of corrections so we get results on more queries.

Fallback chain (each only runs if the previous returned 0 rows):
  1. Raw generated SQL
  2. Regex-fixed SQL
  3. Case-insensitive LIKE rewrite
  4. Fuzzy-matched value correction

All operations are pure (no side effects other than SQLite reads).
"""

from __future__ import annotations

import difflib
import logging
import re
import sqlite3

import pandas as pd

from .database import execute_query

logger = logging.getLogger(__name__)


# ── Regex fixes ───────────────────────────────────────────────────────────────

_FIXES: list[tuple[str, str, int]] = [
    # "FROMt1" → "FROM t1"
    (r"\bFROM\s*t(\d+)", r"FROM t\1", re.IGNORECASE),
    # "FROM 1" → "FROM t1"  (model drops the 't')
    (r"\bFROM\s+(\d+)\b", r"FROM t\1", re.IGNORECASE),
    # "WHEREcol" → "WHERE col"
    (r"\bWHERE(?=[A-Za-z_])", "WHERE ", re.IGNORECASE),
    # "SELECTcol" → "SELECT col"
    (r"\bSELECT(?=[A-Za-z_\*])", "SELECT ", re.IGNORECASE),
    # double spaces
    (r"  +", " ", 0),
]


def fix_sql(sql: str) -> str:
    """Apply all regex corrections to `sql`. Returns the corrected string."""
    result = sql
    for pattern, replacement, flags in _FIXES:
        result = re.sub(pattern, replacement, result, flags=flags)
    return result.strip()


# ── WHERE-clause parser ───────────────────────────────────────────────────────

_WHERE_EQ_PATTERN = re.compile(
    r"""
    SELECT\s+(?P<proj>.+?)\s+
    FROM\s+(?P<table>t\d+)\s+
    WHERE\s+(?P<col>\w+)\s*=\s*['"](?P<val>[^'"]+)['"]
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_where_eq(sql: str):
    """Return regex match groups if sql is a simple col='val' query, else None."""
    return _WHERE_EQ_PATTERN.match(sql.strip())


# ── Fallback strategies ───────────────────────────────────────────────────────

def _try_like(sql: str, conn: sqlite3.Connection) -> tuple[str, pd.DataFrame] | None:
    """
    Rewrite col='val' as lower(col) LIKE '%val%'.
    Helps when the model generates the wrong case or partial value.
    """
    m = _parse_where_eq(sql)
    if not m:
        return None

    proj, table, col, val = m.group("proj", "table", "col", "val")
    like_sql = (
        f"SELECT {proj} FROM {table} "
        f"WHERE lower({col}) LIKE '%{val.lower()}%' COLLATE NOCASE"
    )
    try:
        df = execute_query(like_sql, conn)
        if not df.empty:
            logger.info("LIKE fallback succeeded.")
            return like_sql, df
    except Exception as exc:
        logger.debug("LIKE fallback failed: %s", exc)
    return None


def _try_fuzzy(sql: str, conn: sqlite3.Connection) -> tuple[str, pd.DataFrame] | None:
    """
    Fuzzy-match the literal value against distinct column values.
    Corrects typos and near-matches (e.g. "Jhon" → "John").
    """
    m = _parse_where_eq(sql)
    if not m:
        return None

    proj, table, col, val = m.group("proj", "table", "col", "val")
    try:
        distinct_vals = pd.read_sql_query(
            f"SELECT DISTINCT {col} FROM {table}", conn
        )[col].dropna().astype(str).tolist()
    except Exception as exc:
        logger.debug("Fuzzy fallback: could not fetch distinct values: %s", exc)
        return None

    close = difflib.get_close_matches(val, distinct_vals, n=1, cutoff=0.6)
    if not close:
        return None

    corrected_val = close[0]
    fuzzy_sql = f"SELECT {proj} FROM {table} WHERE {col} = '{corrected_val}'"
    try:
        df = execute_query(fuzzy_sql, conn)
        if not df.empty:
            logger.info("Fuzzy fallback succeeded (corrected '%s' → '%s').", val, corrected_val)
            return fuzzy_sql, df
    except Exception as exc:
        logger.debug("Fuzzy fallback execution failed: %s", exc)
    return None


# ── Public entry point ────────────────────────────────────────────────────────

def run_pipeline(
    raw_sql: str,
    conn: sqlite3.Connection,
) -> tuple[str, pd.DataFrame, str]:
    """
    Apply the full post-processing + fallback chain.

    Returns:
        (final_sql, result_dataframe, strategy_used)

    `strategy_used` is one of:
        "direct", "regex_fix", "like_fallback", "fuzzy_fallback", "failed"

    Never raises — errors are captured and returned as an empty DataFrame.
    """
    # Step 1: raw SQL
    try:
        df = execute_query(raw_sql, conn)
        if not df.empty:
            return raw_sql, df, "direct"
    except Exception as exc:
        logger.debug("Direct execution failed: %s", exc)

    # Step 2: regex-fixed SQL
    fixed = fix_sql(raw_sql)
    if fixed != raw_sql:
        try:
            df = execute_query(fixed, conn)
            if not df.empty:
                return fixed, df, "regex_fix"
        except Exception as exc:
            logger.debug("Regex-fixed execution failed: %s", exc)
    else:
        fixed = raw_sql  # ensure fixed is defined

    # Step 3: LIKE fallback
    result = _try_like(fixed, conn)
    if result:
        return result[0], result[1], "like_fallback"

    # Step 4: fuzzy fallback
    result = _try_fuzzy(fixed, conn)
    if result:
        return result[0], result[1], "fuzzy_fallback"

    # All strategies exhausted
    logger.warning("All fallback strategies failed for SQL: %s", raw_sql)
    return fixed, pd.DataFrame(), "failed"
