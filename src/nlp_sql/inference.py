"""
SQL generation: prompt construction → model inference → raw SQL string.

Fixes vs original notebook:
- Entity extraction is now *actually called* (it was a stub in the demo)
- Generation runs in a background thread with a hard timeout
- Prompt includes dtype + sample info from build_schema_string()
- NER is optional — skipped gracefully if the pipeline isn't loaded
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional

import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast, pipeline

from .config import (
    GENERATION_TIMEOUT,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    NUM_BEAMS,
)

logger = logging.getLogger(__name__)

# Module-level NER pipeline (lazy-loaded on first use, then reused)
_ner_pipeline = None
_NER_MODEL = "dslim/bert-base-NER"


def _get_ner_pipeline(device: torch.device) -> Optional[object]:
    """
    Lazy-load the NER pipeline. Returns None if loading fails (non-fatal).
    """
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline
    try:
        _ner_pipeline = pipeline(
            "ner",
            model=_NER_MODEL,
            grouped_entities=True,
            device=0 if device.type == "cuda" else -1,
        )
        logger.info("NER pipeline loaded (%s).", _NER_MODEL)
    except Exception as exc:
        logger.warning("NER pipeline unavailable (will skip entity extraction): %s", exc)
        _ner_pipeline = None
    return _ner_pipeline


def extract_entities(question: str, device: torch.device) -> list[str]:
    """
    Run BERT-NER on the question and return a de-duplicated list of entity strings.
    Returns an empty list (not an error) if NER is unavailable.
    """
    ner = _get_ner_pipeline(device)
    if ner is None:
        return []
    try:
        results = ner(question)
        words = {e["word"].replace("##", "") for e in results if e.get("word")}
        return sorted(words)
    except Exception as exc:
        logger.warning("NER inference failed: %s", exc)
        return []


def build_prompt(question: str, schema_str: str, device: torch.device) -> str:
    """
    Build the model input string.

    Format:
        [ENT]entity1;entity2[/ENT][SCHEMA]t1(col:TYPE[s1,s2],...)[/SCHEMA]Question: <q>

    Matching the format used during fine-tuning on Spider is critical.
    """
    entities = extract_entities(question, device)
    ent_text = ";".join(entities) if entities else "NONE"
    return f"[ENT]{ent_text}[/ENT][SCHEMA]{schema_str}[/SCHEMA]Question: {question}"


def _run_generation(
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizerFast,
    prompt: str,
    device: torch.device,
) -> str:
    """Inner function executed in a thread so we can apply a timeout."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding="longest",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            max_length=MAX_OUTPUT_LENGTH,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def generate_sql(
    question: str,
    schema_str: str,
    model: BartForConditionalGeneration,
    tokenizer: BartTokenizerFast,
    device: torch.device,
    timeout: int = GENERATION_TIMEOUT,
) -> str:
    """
    Generate a SQL query for `question` given `schema_str`.

    Raises:
        TimeoutError  — if generation takes longer than `timeout` seconds
        RuntimeError  — if the model raises an unexpected error
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not schema_str.strip():
        raise ValueError("Schema string cannot be empty — upload at least one CSV first.")

    prompt = build_prompt(question, schema_str, device)
    logger.debug("Prompt: %s", prompt[:200])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_generation, model, tokenizer, prompt, device)
        try:
            sql = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"SQL generation timed out after {timeout}s. "
                "Try a simpler question or reduce the number of columns."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc

    logger.info("Generated SQL: %s", sql)
    return sql
