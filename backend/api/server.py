#!/usr/bin/env python3
"""
Round 3 live API — serves the 3-pipeline comparison to the React frontend.

Reuses the EXACT pipeline + metric logic from the benchmark
(evaluation.round3_benchmark) so the live demo and the offline benchmark are
identical: same LLM, same temperature/output cap, same token accounting, same
evidence-blind judge, same deterministic numeric-match / support / evidence-quality.

Endpoints:
  GET  /health          — liveness + whether TigerGraph is connected
  GET  /stats/session   — session counters (queries run, tokens saved, avg reduction)
  POST /query/compare   — run LLM-Only vs Traditional RAG vs Optimized GraphRAG

Run:
  uvicorn backend.api.server:app --reload --port 8000
"""

import logging
import threading
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse the benchmark's pipeline runners + metric helpers verbatim (single source
# of truth — the live demo cannot diverge from the reported benchmark numbers).
from evaluation.round3_benchmark import (
    run_llm_only, run_rag, run_graphrag,
    judge, numeric_match, cost, _key_nums, _gold_tickers, _doc_tickers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="TokenNinja Round 3 — 3-Pipeline API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # local demo — frontend dev server on :5173
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy singletons: connect to TigerGraph + build pipelines once, on first use.
#    Keeps the server bootable even if TG is momentarily unreachable (the error
#    then surfaces per-request instead of crashing startup).
_lock = threading.Lock()
_pipes = {"tg": None, "rag": None, "gr": None}

# ── Session counters (in-memory; reset on restart) ───────────────────────────
_session = {"total_queries": 0, "total_tokens_saved": 0, "reductions": []}


def _get_pipes():
    if _pipes["rag"] is None:
        with _lock:
            if _pipes["rag"] is None:
                from backend.graph.tigergraph_client import TigerGraphClient
                from backend.rag.basic_rag_tg import BasicRAGTigerGraph
                from backend.rag.graph_rag_tg import GraphRAGTG
                logger.info("Connecting to TigerGraph + building pipelines...")
                tg = TigerGraphClient().connect()
                _pipes["tg"] = tg
                _pipes["rag"] = BasicRAGTigerGraph(tg)
                _pipes["gr"] = GraphRAGTG(tg)
                logger.info("Pipelines ready.")
    return _pipes["rag"], _pipes["gr"]


class CompareRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = ""      # optional reference → enables grade/pass/numeric-match
    run_judge: bool = True


def _finish(pipe: dict, question: str, reference: str, run_judge: bool, gold: set) -> dict:
    """Attach the same deterministic + judged metrics the benchmark computes."""
    ref = (reference or "").strip()
    # Deterministic support + evidence-quality (reproducible, not LLM-judged) — always available.
    ans_nums = _key_nums(pipe["answer"])
    ev_nums = _key_nums(pipe.get("evidence", ""))
    support = 1.0 if (ans_nums and (ans_nums & ev_nums)) else (0.0 if ans_nums else None)
    got = _doc_tickers(pipe["citations"])
    cite_recall = round(len(gold & got) / len(gold), 3) if gold else None
    pipe["supported"] = (support == 1.0) if support is not None else None
    pipe["evidence_quality"] = round(0.5 * (cite_recall or 0) + 0.5 * (support if support is not None else 0), 3)
    pipe["citation_recall"] = cite_recall
    pipe["cost"] = cost(pipe["prompt_tokens"], pipe["output_tokens"])
    # Judged correctness (evidence-blind, reference-only) — only if a reference is given.
    if ref and run_judge:
        g, pas = judge(question, ref, pipe["answer"])
        pipe["grade"], pipe["pass"] = g, pas
        pipe["numeric_match"] = numeric_match(ref, pipe["answer"]) if _key_nums(ref) else None
    else:
        pipe["grade"] = pipe["pass"] = pipe["numeric_match"] = None
    return pipe


@app.get("/health")
def health():
    return {"status": "ok", "tigergraph_connected": _pipes["rag"] is not None}


@app.get("/stats/session")
def stats_session():
    reds = _session["reductions"]
    return {
        "total_queries": _session["total_queries"],
        "total_tokens_saved": _session["total_tokens_saved"],
        "avg_token_reduction_pct": round(sum(reds) / len(reds), 1) if reds else 0,
    }


@app.post("/query/compare")
def query_compare(req: CompareRequest):
    q = (req.question or "").strip()
    if not q:
        return {"error": "empty question"}
    ref = (req.ground_truth or "").strip()
    gold = _gold_tickers(ref) if ref else set()

    try:
        rag, gr = _get_pipes()
    except Exception as e:
        logger.exception("pipeline init failed")
        return {"error": f"TigerGraph connection failed: {e}"}

    try:
        llm = _finish(run_llm_only(q), q, ref, req.run_judge, gold)
        rg = _finish(run_rag(rag, q), q, ref, req.run_judge, gold)
        grg = _finish(run_graphrag(gr, q), q, ref, req.run_judge, gold)
    except Exception as e:
        logger.exception("pipeline run failed")
        return {"error": f"pipeline error: {e}"}

    rag_tok = rg["total_inference"] or 1
    gr_tok = grg["total_inference"]
    token_red = round((rag_tok - gr_tok) / rag_tok * 100, 1)
    rag_ctx = rg["context_tokens"] or 1
    ctx_red = round((rag_ctx - grg["context_tokens"]) / rag_ctx * 100, 1)

    # session counters
    _session["total_queries"] += 1
    _session["total_tokens_saved"] += max(0, rag_tok - gr_tok)
    _session["reductions"].append(token_red)

    validity_ok = None
    if ref and req.run_judge:
        validity_ok = (llm["grade"] <= rg["grade"] <= grg["grade"])

    return {
        "question": q,
        "reference": ref,
        "graded": bool(ref and req.run_judge),
        "llm_only": llm,
        "rag": rg,
        "graphrag": grg,
        "token_reduction_pct": token_red,
        "context_reduction_pct": ctx_red,
        "validity_ordering_ok": validity_ok,
    }
