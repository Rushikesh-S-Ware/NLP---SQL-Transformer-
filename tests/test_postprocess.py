"""
Tests for src/nlp_sql/postprocess.py

Covers:
- Each individual regex fix
- LIKE fallback
- Fuzzy value matching
- Full pipeline orchestration (all strategies)
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.nlp_sql.postprocess import fix_sql, run_pipeline


# ── fix_sql ────────────────────────────────────────────────────────────────────

class TestFixSql:

    def test_fixes_fromt1_no_space(self):
        assert "FROM t1" in fix_sql("SELECT * FROMt1")

    def test_fixes_from_bare_number(self):
        result = fix_sql("SELECT * FROM 1")
        assert "FROM t1" in result

    def test_fixes_from_bare_number_2(self):
        result = fix_sql("SELECT name FROM 2")
        assert "FROM t2" in result

    def test_fixes_where_no_space(self):
        result = fix_sql("SELECT * FROM t1 WHEREage > 25")
        assert "WHERE age" in result

    def test_fixes_select_no_space(self):
        result = fix_sql("SELECTname FROM t1")
        assert "SELECT name" in result

    def test_collapses_double_spaces(self):
        result = fix_sql("SELECT  *  FROM  t1")
        assert "  " not in result

    def test_no_change_on_clean_sql(self):
        sql = "SELECT name FROM t1 WHERE age > 25"
        assert fix_sql(sql) == sql

    def test_handles_multiword_projection(self):
        result = fix_sql("SELECT name, age FROMt1")
        assert "FROM t1" in result

    def test_strips_trailing_whitespace(self):
        result = fix_sql("SELECT * FROM t1   ")
        assert not result.endswith(" ")


# ── run_pipeline (integration) ─────────────────────────────────────────────────

@pytest.fixture
def people_conn():
    """Employees table with mixed-case names for testing case/fuzzy fallbacks."""
    conn = sqlite3.connect(":memory:")
    df = pd.DataFrame({
        "id":   [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "dept": ["Engineering", "Marketing", "Engineering", "HR"],
        "age":  [30, 25, 35, 28],
    })
    df.to_sql("t1", conn, index=False, if_exists="replace")
    yield conn
    conn.close()


class TestRunPipeline:

    def test_direct_success(self, people_conn):
        sql = "SELECT name FROM t1 WHERE dept = 'Engineering'"
        final_sql, df, strategy = run_pipeline(sql, people_conn)
        assert strategy == "direct"
        assert len(df) == 2
        assert "Alice" in df["name"].values

    def test_regex_fix_applied(self, people_conn):
        # Missing space after FROM — should be fixed by regex step
        sql = "SELECT name FROMt1 WHERE dept = 'HR'"
        final_sql, df, strategy = run_pipeline(sql, people_conn)
        assert strategy in ("regex_fix", "direct")
        assert not df.empty

    def test_like_fallback_case_mismatch(self, people_conn):
        # Wrong case — direct will fail, LIKE fallback should match
        sql = "SELECT name FROM t1 WHERE dept = 'engineering'"
        final_sql, df, strategy = run_pipeline(sql, people_conn)
        assert strategy in ("like_fallback", "direct")
        assert not df.empty

    def test_fuzzy_fallback_typo(self, people_conn):
        # "Enginering" (typo) — fuzzy should match "Engineering"
        sql = "SELECT name FROM t1 WHERE dept = 'Enginering'"
        final_sql, df, strategy = run_pipeline(sql, people_conn)
        assert strategy in ("fuzzy_fallback", "like_fallback", "direct")
        assert not df.empty

    def test_returns_failed_on_garbage_sql(self, people_conn):
        sql = "SELECT blah FROM nonexistent_table_xyz"
        final_sql, df, strategy = run_pipeline(sql, people_conn)
        assert strategy == "failed"
        assert df.empty

    def test_returns_tuple_always(self, people_conn):
        """run_pipeline must never raise — always returns a 3-tuple."""
        result = run_pipeline("THIS IS NOT SQL AT ALL", people_conn)
        assert len(result) == 3

    def test_aggregate_query_direct(self, people_conn):
        sql = "SELECT dept, COUNT(*) AS cnt FROM t1 GROUP BY dept"
        final_sql, df, strategy = run_pipeline(sql, people_conn)
        assert strategy == "direct"
        assert "cnt" in df.columns

    def test_empty_string_returns_failed(self, people_conn):
        final_sql, df, strategy = run_pipeline("", people_conn)
        assert strategy == "failed"
        assert df.empty
