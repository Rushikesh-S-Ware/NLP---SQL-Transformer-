"""
Model loading with automatic checkpoint resolution.

Priority order for finding the model:
  1. HF_MODEL_ID environment variable  → load from HuggingFace Hub
  2. LOCAL_CHECKPOINT_DIR              → already extracted locally
  3. BUNDLED_ZIP                       → extract the repo's zip on first run

This means the app works in all three environments:
  - Local dev    (extracted checkpoint)
  - HF Space     (set HF_MODEL_ID secret)
  - First-run    (auto-extracts zip)
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast

from .config import BUNDLED_ZIP, HF_MODEL_ID, LOCAL_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def _extract_bundled_zip(zip_path: Path, dest: Path) -> None:
    """
    Handle the double-zipped checkpoint (Model_Checkpoint.zip.zip).
    Extracts until we reach the model directory.
    """
    logger.info("Extracting checkpoint from %s …", zip_path)
    dest.mkdir(parents=True, exist_ok=True)

    current = zip_path
    tmp_dir = dest.parent / "_zip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Unzip recursively until no more zip layers
    for _ in range(3):
        if not zipfile.is_zipfile(current):
            break
        with zipfile.ZipFile(current, "r") as zf:
            zf.extractall(tmp_dir)
        # Find the extracted content
        extracted = list(tmp_dir.iterdir())
        if len(extracted) == 1 and extracted[0].is_file() and zipfile.is_zipfile(extracted[0]):
            current = extracted[0]           # another zip layer
        elif len(extracted) == 1 and extracted[0].is_dir():
            shutil.copytree(extracted[0], dest, dirs_exist_ok=True)
            shutil.rmtree(tmp_dir)
            return
        else:
            # Multiple files — assume this is the model directory contents
            for item in extracted:
                target = dest / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
            shutil.rmtree(tmp_dir)
            return

    shutil.rmtree(tmp_dir, ignore_errors=True)
    raise RuntimeError(
        f"Could not extract a valid checkpoint from {zip_path}. "
        "Please extract it manually to the 'checkpoint/' directory."
    )


def _resolve_checkpoint() -> str:
    """
    Return a model identifier (HF Hub ID or local path string) that
    BartForConditionalGeneration.from_pretrained() can consume.
    """
    # 1. HuggingFace Hub via env var
    if HF_MODEL_ID:
        logger.info("Using HuggingFace Hub model: %s", HF_MODEL_ID)
        return HF_MODEL_ID

    # 2. Already extracted locally
    if LOCAL_CHECKPOINT_DIR.exists() and any(LOCAL_CHECKPOINT_DIR.iterdir()):
        logger.info("Using local checkpoint at %s", LOCAL_CHECKPOINT_DIR)
        return str(LOCAL_CHECKPOINT_DIR)

    # 3. Extract from bundled zip
    if BUNDLED_ZIP.exists():
        _extract_bundled_zip(BUNDLED_ZIP, LOCAL_CHECKPOINT_DIR)
        if LOCAL_CHECKPOINT_DIR.exists() and any(LOCAL_CHECKPOINT_DIR.iterdir()):
            logger.info("Checkpoint extracted to %s", LOCAL_CHECKPOINT_DIR)
            return str(LOCAL_CHECKPOINT_DIR)

    raise FileNotFoundError(
        "No model checkpoint found.\n"
        "Options:\n"
        "  • Set the HF_MODEL_ID environment variable to your HuggingFace model ID.\n"
        "  • Extract Model_Checkpoint.zip.zip into a folder named 'checkpoint/'.\n"
        "  • Run the training notebook and save the output to 'checkpoint/'."
    )


def load_model(
    device: torch.device | None = None,
) -> tuple[BartForConditionalGeneration, BartTokenizerFast]:
    """
    Load and return (model, tokenizer).

    The model is moved to `device` (defaults to CUDA if available, else CPU).
    Call this once at startup and cache the result (e.g. with @st.cache_resource).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _resolve_checkpoint()
    logger.info("Loading tokenizer from %s …", checkpoint)
    tokenizer = BartTokenizerFast.from_pretrained(checkpoint)

    logger.info("Loading model from %s on %s …", checkpoint, device)
    model = BartForConditionalGeneration.from_pretrained(checkpoint)
    model = model.to(device)
    model.eval()

    return model, tokenizer
