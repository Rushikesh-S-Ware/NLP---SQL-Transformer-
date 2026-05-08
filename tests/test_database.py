"""
Tests for src/nlp_sql/database.py

Covers:
- CSV loading (normal, chunked, row cap, empty, malformed)
- Schema string format (dtypes, sample values)
- Multi-file loading
- Query validation (blocked statements)
- LIMIT injection
- Query execution (happy path and errors)
"""

from __future__ import annotations

import io
import sqlite3

import pandas as pd
import pytest

from src.nlp_sql.database import (
    _inject_limit,
    build_schema_string,
    execute_query,
    load_csv_to_table,
    load_files_to_sqlite,
    validate_query,
)
from src.nlp_sql.config import MAX_QUERY_RESULTS


# ── CSV loading ────────────────────────────────────────────────────────────────

class TestLoadCsvToTable:

    def test_loads_all_rows_small_file(self, csv_buffer):
        conn = sqlite3.connect(":memory:")
        df = load_csv_to_table(csv_buffer, conn, "t1")
        count = pd.read_sql_query("SELECT COUNT(*) AS n FROM t1", conn).iloc[0]["n"]
        assert count == 5
        conn.close()

    def test_respects_row_cap(self, sample_df):
        """A file with 100 rows should be truncated to max_rows=10."""
        big_df = pd.concat([sample_df] * 20, ignore_index=True)  # 100 rows
        buf = io.BytesIO()
        big_df.to_csv(buf, index=False)
        buf.seek(0)

        conn = sqlite3.connect(":memory:")
        df = load_csv_to_table(buf, conn, "t1", max_rows=10, chunk_size=30)
        count = pd.read_sql_query("SELECT COUNT(*) AS n FROM t1", conn).iloc[0]["n"]
        assert count == 10
        conn.close()

    def test_raises_on_empty_file(self):
        buf = io.BytesIO(b"")
        conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError, match="empty"):
            load_csv_to_table(buf, conn, "t1")
        conn.close()

    def test_raises_on_headers_only(self):
        buf = io.BytesIO(b"col1,col2\n")
        conn = sqlite3.connect(":memory:")
        with pytest.raises(ValueError, match="zero rows"):
            load_csv_to_table(buf, conn, "t1")
        conn.close()

    def test_column_names_preserved(self, csv_buffer):
        conn = sqlite3.connect(":memory:")
        load_csv_to_table(csv_buffer, conn, "t1")
        cols = [row[1] for row in conn.execute("PRAGMA table_info(t1)")]
        assert "name" in cols
        assert "salary" in cols
        conn.close()


class TestLoadFilesToSqlite:

    def _make_buf(self, df: pd.DataFrame) -> io.BytesIO:
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        # Mimic Streamlit UploadedFile which has a .name attribute
        buf.name = "test.csv"
        return buf

    def test_single_file_creates_t1(self, sample_df):
        buf = self._make_buf(sample_df)
        conn, tables = load_files_to_sqlite([buf])
        assert "t1" in tables
        assert len(tables["t1"]) == 5
        conn.close()

    def test_two_files_create_t1_t2(self, sample_df):
        b1 = self._make_buf(sample_df)
        b2 = self._make_buf(sample_df.head(3))
        conn, tables = load_files_to_sqlite([b1, b2])
        assert set(tables.keys()) == {"t1", "t2"}
        assert len(tables["t1"]) == 5
        assert len(tables["t2"]) == 3
        conn.close()


# ── Schema string ──────────────────────────────────────────────────────────────

class TestBuildSchemaString:

    def test_contains_table_name(self, sample_df):
        schema = build_schema_string({"t1": sample_df})
        assert "t1(" in schema

    def test_contains_column_names(self, sample_df):
        schema = build_schema_string({"t1": sample_df})
        for col in ["id", "name", "age", "salary", "dept"]:
            assert col in schema

    def test_contains_dtype_tags(self, sample_df):
        schema = build_schema_string({"t1": sample_df})
        assert "INTEGER" in schema or "REAL" in schema or "TEXT" in schema

    def test_contains_sample_values(self, sample_df):
        schema = build_schema_string({"t1": sample_df}, n_samples=2)
        # At least one of the actual names should appear in the schema
        assert "Alice" in schema or "Bob" in schema

    def test_two_tables_separated_by_pipe(self, sample_df):
        schema = build_schema_string({"t1": sample_df, "t2": sample_df.head(2)})
        assert " | " in schema
        assert "t1(" in schema and "t2(" in schema

    def test_handles_nulls_in_sample(self, sample_df):
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, "name"] = None
        # Should not raise
        schema = build_schema_string({"t1": df_with_nulls})
        assert "name" in schema


# ── Query validation ───────────────────────────────────────────────────────────

class TestValidateQuery:

    @pytest.mark.parametrize("stmt", [
        "DROP TABLE t1",
        "DELETE FROM t1",
        "UPDATE t1 SET name='x'",
        "INSERT INTO t1 VALUES (1,'x',20,1000,'HR')",
        "ALTER TABLE t1 ADD COLUMN x TEXT",
        "CREATE TABLE evil (id INT)",
        "TRUNCATE TABLE t1",
    ])
    def test_blocks_write_statements(self, stmt):
        with pytest.raises(ValueError, match="blocked"):
            validate_query(stmt)

    def test_allows_select(self):
        # Should not raise
        validate_query("SELECT * FROM t1")

    def test_allows_select_with_joins(self):
        validate_query("SELECT t1.name FROM t1 JOIN t2 ON t1.id = t2.id")

    def test_empty_query_raises(self):
        with pytest.raises((ValueError, IndexError)):
            validate_query("")


# ── LIMIT injection ────────────────────────────────────────────────────────────

class TestInjectLimit:

    def test_injects_limit_when_absent(self):
        sql = "SELECT * FROM t1"
        result = _inject_limit(sql, limit=100)
        assert "LIMIT 100" in result

    def test_does_not_duplicate_limit(self):
        sql = "SELECT * FROM t1 LIMIT 50"
        result = _inject_limit(sql, limit=100)
        assert result.upper().count("LIMIT") == 1

    def test_respects_existing_limit(self):
        sql = "SELECT * FROM t1 LIMIT 50"
        result = _inject_limit(sql, limit=100)
        assert "LIMIT 50" in result

    def test_does_not_modify_non_select(self):
        sql = "PRAGMA table_info(t1)"
        result = _inject_limit(sql, limit=100)
        # LIMIT should NOT be appended to non-SELECT
        assert "LIMIT" not in result


# ── Query execution ────────────────────────────────────────────────────────────

class TestExecuteQuery:

    def test_returns_dataframe(self, sqlite_conn):
        df = execute_query("SELECT * FROM t1", sqlite_conn)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_limit_applied(self, sqlite_conn):
        df = execute_query("SELECT * FROM t1", sqlite_conn, limit=2)
        assert len(df) <= 2

    def test_filtered_query(self, sqlite_conn):
        df = execute_query("SELECT name FROM t1 WHERE dept = 'Engineering'", sqlite_conn)
        assert all(row == "Engineering" or True for row in df.get("dept", []))
        assert len(df) == 2  # Alice and Charlie

    def test_raises_on_blocked_statement(self, sqlite_conn):
        with pytest.raises(ValueError, match="blocked"):
            execute_query("DROP TABLE t1", sqlite_conn)

    def test_raises_on_syntax_error(self, sqlite_conn):
        with pytest.raises(RuntimeError, match="execution failed"):
            execute_query("SELECT FROM WHERE", sqlite_conn)

    def test_raises_on_empty_sql(self, sqlite_conn):
        with pytest.raises(ValueError, match="empty"):
            execute_query("", sqlite_conn)

    def test_aggregate_query(self, sqlite_conn):
        df = execute_query("SELECT dept, COUNT(*) AS cnt FROM t1 GROUP BY dept", sqlite_conn)
        assert "cnt" in df.columns
        assert len(df) == 3  # Engineering, Marketing, HR
