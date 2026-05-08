"""
Shared pytest fixtures for the NLP-SQL Transformer test suite.
"""

from __future__ import annotations

import io
import sqlite3
from unittest.mock import MagicMock

import pandas as pd
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore


# ---------- Sample DataFrames ----------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age":    [30, 25, 35, 28, 22],
        "salary": [70000.0, 55000.0, 90000.0, 62000.0, 48000.0],
        "dept":   ["Engineering", "Marketing", "Engineering", "HR", "Marketing"],
    })


@pytest.fixture
def csv_buffer(sample_df):
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


@pytest.fixture
def sqlite_conn(sample_df):
    conn = sqlite3.connect(":memory:")
    sample_df.to_sql("t1", conn, index=False, if_exists="replace")
    yield conn
    conn.close()


@pytest.fixture
def two_table_conn():
    conn = sqlite3.connect(":memory:")
    employees = pd.DataFrame({
        "emp_id":  [1, 2, 3],
        "name":    ["Alice", "Bob", "Charlie"],
        "dept_id": [10, 20, 10],
    })
    departments = pd.DataFrame({
        "dept_id": [10, 20],
        "dept":    ["Engineering", "Marketing"],
    })
    employees.to_sql("t1", conn, index=False, if_exists="replace")
    departments.to_sql("t2", conn, index=False, if_exists="replace")
    yield conn
    conn.close()


# ---------- Mock model/tokenizer (skipped if torch absent) ----------

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    encoded = MagicMock()
    encoded.to.return_value = encoded
    encoded.__getitem__ = lambda self, key: MagicMock()
    tok.return_value = encoded
    tok.decode.return_value = "SELECT * FROM t1"
    tok.pad_token_id = 1
    return tok


@pytest.fixture
def mock_model():
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    model = MagicMock()
    model.generate.return_value = torch.tensor([[0, 1, 2, 3]])
    return model


@pytest.fixture
def mock_device():
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    return torch.device("cpu")
