#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=[runs[0].stem])
rows = [json.loads(l) for p in runs if p.stem in chosen
        for l in p.open(encoding="utf-8") if l.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

c = st.columns(5)
c[0].metric("spans", len(df))
requests = df[df["name"] == "service.request"]
total_cost = requests.get("cost_usd", pd.Series(0.0, index=requests.index)).fillna(0).sum()
c[1].metric("total cost", f"${total_cost:.4f}")
llm = df[df["name"] == "llm.call"]
c[2].metric("model calls", len(llm))
if len(requests):
    cache_rate = requests.get(
        "cached", pd.Series(False, index=requests.index)
    ).fillna(False).mean()
elif len(llm):
    cache_rate = llm.get(
        "cached", pd.Series(False, index=llm.index)
    ).fillna(False).mean()
else:
    cache_rate = 0.0
c[3].metric("cache hit rate", f"{cache_rate:.0%}")
attempts = df[df["name"] == "http.ask"]
error_count = int((attempts.get("status") == "error").sum())
c[4].metric("error rate", f"{error_count / max(1, len(attempts)):.1%}")

st.subheader("Latency by stage")
duration_df = df[df.get("duration_ms", pd.Series(index=df.index, dtype=float)).notna()]
if not duration_df.empty:
    stage = (
        duration_df.groupby("name")["duration_ms"]
        .agg(n="count", p50="median", p95=lambda s: s.quantile(0.95), total="sum")
        .sort_values("total", ascending=False)
    )
    st.dataframe(stage, use_container_width=True)
    stage_rows = duration_df[["ts", "name", "duration_ms"]].copy()
    stage_rows = stage_rows.set_index("ts")
    st.line_chart(
        stage_rows.pivot_table(
            index="ts", columns="name", values="duration_ms", aggfunc="mean"
        ).sort_index()
    )
else:
    st.info("Selected traces contain no duration measurements.")

st.subheader("Cost over time")
if not requests.empty and "cost_usd" in requests:
    cum = requests.sort_values("ts").assign(
        cum=lambda d: d["cost_usd"].fillna(0).cumsum()
    )
    st.line_chart(cum.set_index("ts")["cum"])
else:
    st.info("Selected runs contain no service.request cost events.")

st.subheader("Errors")
errs = df[df.get("status") == "error"]
st.dataframe(errs[["ts", "name", "error"]] if len(errs) else pd.DataFrame(),
             use_container_width=True)

st.subheader("Refusal-rate alert")
requests = requests.sort_values("ts")
if len(requests) >= 10:
    half = max(1, len(requests) // 2)
    baseline = float(requests.iloc[:half]["refused"].fillna(False).mean())
    recent = float(requests.iloc[-half:]["refused"].fillna(False).mean())
    if baseline > 0 and recent >= 2 * baseline:
        st.error(
            f"Refusal rate doubled ({baseline:.0%} to {recent:.0%}). "
            "Action: check index build/refresh logs and retrieval-stage traces, "
            "then run the golden retrieval gate before restoring traffic."
        )
    else:
        st.success(
            f"No doubling detected. Earlier {baseline:.0%}; recent {recent:.0%}."
        )
else:
    st.info("Need at least 10 service.request events to compare refusal rates.")
