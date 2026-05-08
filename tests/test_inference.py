"""
Tests for src/nlp_sql/inference.py

The model and tokenizer are always mocked — these tests verify the
orchestration logic (prompt construction, timeout handling, error paths)
without requiring the actual 500MB BART checkpoint.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

from src.nlp_sql.inference import build_prompt, extract_entities, generate_sql  # noqa: E402


# ── build_prompt ───────────────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_contains_schema(self, mock_device):
        schema = "t1(name:TEXT[Alice,Bob], age:INTEGER[30,25])"
        prompt = build_prompt("Who is oldest?", schema, mock_device)
        assert schema in prompt

    def test_contains_question(self, mock_device):
        prompt = build_prompt("How many employees?", "t1(id:INTEGER[])", mock_device)
        assert "How many employees?" in prompt

    def test_has_ent_tags(self, mock_device):
        prompt = build_prompt("Show me Alice", "t1(name:TEXT[])", mock_device)
        assert "[ENT]" in prompt
        assert "[/ENT]" in prompt

    def test_has_schema_tags(self, mock_device):
        prompt = build_prompt("Count rows", "t1(id:INTEGER[])", mock_device)
        assert "[SCHEMA]" in prompt
        assert "[/SCHEMA]" in prompt

    def test_none_entities_shown_as_none(self, mock_device):
        # When NER is unavailable, entities should be "NONE"
        with patch("src.nlp_sql.inference._get_ner_pipeline", return_value=None):
            prompt = build_prompt("What is the total?", "t1(x:INTEGER[])", mock_device)
        assert "[ENT]NONE[/ENT]" in prompt

    def test_prompt_order(self, mock_device):
        """[ENT]…[/ENT] must come before [SCHEMA]…[/SCHEMA] Question:"""
        with patch("src.nlp_sql.inference._get_ner_pipeline", return_value=None):
            prompt = build_prompt("test", "t1(x:TEXT[])", mock_device)
        ent_pos    = prompt.index("[ENT]")
        schema_pos = prompt.index("[SCHEMA]")
        q_pos      = prompt.index("Question:")
        assert ent_pos < schema_pos < q_pos


# ── extract_entities ───────────────────────────────────────────────────────────

class TestExtractEntities:

    def test_returns_list(self, mock_device):
        with patch("src.nlp_sql.inference._get_ner_pipeline", return_value=None):
            result = extract_entities("Find Alice in Engineering", mock_device)
        assert isinstance(result, list)

    def test_empty_when_ner_unavailable(self, mock_device):
        with patch("src.nlp_sql.inference._get_ner_pipeline", return_value=None):
            result = extract_entities("What is the revenue?", mock_device)
        assert result == []

    def test_ner_result_parsed(self, mock_device):
        mock_ner = MagicMock(return_value=[
            {"word": "Alice", "entity_group": "PER", "score": 0.99},
            {"word": "Google", "entity_group": "ORG", "score": 0.95},
        ])
        with patch("src.nlp_sql.inference._get_ner_pipeline", return_value=mock_ner):
            result = extract_entities("Alice works at Google", mock_device)
        assert "Alice" in result
        assert "Google" in result

    def test_deduplicates_entities(self, mock_device):
        mock_ner = MagicMock(return_value=[
            {"word": "Alice", "entity_group": "PER", "score": 0.99},
            {"word": "Alice", "entity_group": "PER", "score": 0.95},
        ])
        with patch("src.nlp_sql.inference._get_ner_pipeline", return_value=mock_ner):
            result = extract_entities("Alice and Alice", mock_device)
        assert result.count("Alice") == 1


# ── generate_sql ───────────────────────────────────────────────────────────────

class TestGenerateSql:

    def _patch_generation(self, sql_output: str):
        """Context manager that patches _run_generation to return sql_output."""
        return patch(
            "src.nlp_sql.inference._run_generation",
            return_value=sql_output,
        )

    def test_returns_string(self, mock_model, mock_tokenizer, mock_device):
        with self._patch_generation("SELECT * FROM t1"):
            result = generate_sql(
                "Show all rows",
                "t1(id:INTEGER[])",
                mock_model,
                mock_tokenizer,
                mock_device,
            )
        assert isinstance(result, str)
        assert "SELECT" in result

    def test_raises_on_empty_question(self, mock_model, mock_tokenizer, mock_device):
        with pytest.raises(ValueError, match="empty"):
            generate_sql("", "t1(id:INTEGER[])", mock_model, mock_tokenizer, mock_device)

    def test_raises_on_empty_schema(self, mock_model, mock_tokenizer, mock_device):
        with pytest.raises(ValueError, match="empty"):
            generate_sql("Count rows", "", mock_model, mock_tokenizer, mock_device)

    def test_raises_on_whitespace_question(self, mock_model, mock_tokenizer, mock_device):
        with pytest.raises(ValueError, match="empty"):
            generate_sql("   ", "t1(id:INTEGER[])", mock_model, mock_tokenizer, mock_device)

    def test_timeout_raises_timeout_error(self, mock_model, mock_tokenizer, mock_device):
        def slow(*args, **kwargs):
            import time
            time.sleep(10)
            return "SELECT 1"

        with patch("src.nlp_sql.inference._run_generation", side_effect=slow):
            with pytest.raises(TimeoutError):
                generate_sql(
                    "Question",
                    "t1(id:INTEGER[])",
                    mock_model,
                    mock_tokenizer,
                    mock_device,
                    timeout=1,      # 1-second timeout
                )

    def test_model_exception_wrapped(self, mock_model, mock_tokenizer, mock_device):
        with patch(
            "src.nlp_sql.inference._run_generation",
            side_effect=RuntimeError("CUDA OOM"),
        ):
            with pytest.raises(RuntimeError, match="inference failed"):
                generate_sql(
                    "Question",
                    "t1(id:INTEGER[])",
                    mock_model,
                    mock_tokenizer,
                    mock_device,
                )
