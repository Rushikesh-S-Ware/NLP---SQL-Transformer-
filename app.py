"""
NLP → SQL Transformer — Streamlit App

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → GitHub Actions auto-deploys to HuggingFace Spaces.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="NLP → SQL Transformer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Lazy imports (avoid loading torch/transformers until needed) ───────────────
from src.nlp_sql.database import build_schema_string, load_files_to_sqlite
from src.nlp_sql.inference import generate_sql
from src.nlp_sql.model import load_model
from src.nlp_sql.postprocess import run_pipeline


# ── Model loading (cached — loads once per server process) ────────────────────
@st.cache_resource(show_spinner="Loading model… this takes ~30s on first run.")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(device)
    return model, tokenizer, device


# ── Session state initialisation ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of (question, sql, strategy, n_rows)
if "conn" not in st.session_state:
    st.session_state.conn = None
if "tables" not in st.session_state:
    st.session_state.tables = {}
if "schema_str" not in st.session_state:
    st.session_state.schema_str = ""


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 NLP → SQL")
    st.caption("Ask questions about your CSV data in plain English.")
    st.divider()

    st.subheader("📂 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Each file becomes a table: t1, t2, t3 …",
    )

    max_rows = st.slider(
        "Max rows per table",
        min_value=1_000,
        max_value=100_000,
        value=50_000,
        step=1_000,
        help="Larger values use more memory. Reduce if the app slows down.",
    )

    load_btn = st.button("Load Files", type="primary", use_container_width=True)

    if load_btn and uploaded_files:
        with st.spinner("Loading CSVs into database…"):
            try:
                if st.session_state.conn:
                    st.session_state.conn.close()
                conn, tables = load_files_to_sqlite(uploaded_files, max_rows=max_rows)
                st.session_state.conn = conn
                st.session_state.tables = tables
                st.session_state.schema_str = build_schema_string(tables)
                st.session_state.history = []
                st.success(f"Loaded {len(tables)} table(s).")
            except Exception as e:
                st.error(f"Failed to load files: {e}")

    elif load_btn and not uploaded_files:
        st.warning("Please upload at least one CSV file first.")

    # Schema preview
    if st.session_state.tables:
        st.divider()
        st.subheader("📋 Schema")
        for tbl, df in st.session_state.tables.items():
            with st.expander(f"{tbl} — {len(df):,} rows, {len(df.columns)} cols"):
                st.dataframe(df.head(5), use_container_width=True)

    st.divider()
    st.subheader("⚙️ Model")
    device_label = "GPU 🚀" if torch.cuda.is_available() else "CPU"
    st.caption(f"Running on: **{device_label}**")

    if st.button("Load Model", use_container_width=True):
        with st.spinner("Loading model…"):
            try:
                get_model()
                st.success("Model ready.")
            except Exception as e:
                st.error(f"Model load failed: {e}")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Ask Your Data a Question")
st.caption("Type a natural-language question. The model generates and runs SQL for you.")

# Question input
col_q, col_go = st.columns([5, 1])
with col_q:
    question = st.text_input(
        "Your question",
        placeholder='e.g. "Who are the top 5 customers by total sales?"',
        label_visibility="collapsed",
    )
with col_go:
    submit = st.button("Ask →", type="primary", use_container_width=True)

# ── Query execution ───────────────────────────────────────────────────────────
if submit:
    if not question.strip():
        st.warning("Please type a question.")
    elif not st.session_state.conn:
        st.warning("Upload and load CSV files first (use the sidebar).")
    else:
        col_sql, col_info = st.columns([3, 1])

        with st.spinner("Generating SQL…"):
            try:
                model, tokenizer, device = get_model()
            except Exception as e:
                st.error(f"Model not available: {e}")
                st.stop()

            t0 = time.perf_counter()
            try:
                raw_sql = generate_sql(
                    question,
                    st.session_state.schema_str,
                    model,
                    tokenizer,
                    device,
                )
            except TimeoutError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                st.error(f"Generation error: {e}")
                st.stop()

            gen_ms = (time.perf_counter() - t0) * 1000

        with st.spinner("Running query…"):
            final_sql, result_df, strategy = run_pipeline(
                raw_sql, st.session_state.conn
            )

        # ── Results layout ─────────────────────────────────────────────────
        with col_sql:
            st.subheader("Generated SQL")
            st.code(final_sql, language="sql")

        with col_info:
            st.subheader("Stats")
            st.metric("Generation", f"{gen_ms:.0f} ms")
            st.metric("Rows returned", len(result_df))
            strategy_labels = {
                "direct":        "✅ Direct",
                "regex_fix":     "🔧 Regex fix",
                "like_fallback": "🔍 LIKE fallback",
                "fuzzy_fallback":"🔮 Fuzzy match",
                "failed":        "❌ No results",
            }
            st.metric("Strategy", strategy_labels.get(strategy, strategy))

        st.subheader("Results")
        if result_df.empty:
            st.info(
                "No rows returned. The model may have generated an incorrect query. "
                "Try rephrasing your question or check the SQL above."
            )
        else:
            st.dataframe(result_df, use_container_width=True)

        # Save to history
        st.session_state.history.append(
            (question, final_sql, strategy, len(result_df))
        )

# ── Query history ─────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander(f"📜 Query History ({len(st.session_state.history)} queries)"):
        for i, (q, sql, strat, n) in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Q{len(st.session_state.history) - i}:** {q}")
            st.code(sql, language="sql")
            st.caption(f"Strategy: {strat} · Rows: {n}")
            st.divider()

    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:grey; margin-top:2rem; font-size:0.8rem'>"
    "NLP-SQL Transformer · Fine-tuned BART on Spider · George Mason University"
    "</div>",
    unsafe_allow_html=True,
)
