#!/usr/bin/env python3
"""
Round 3 benchmark — LLM-Only vs Traditional RAG vs Optimized GraphRAG.

Implements the Round 3 controlled-evaluation + token-accounting requirements:
 • same generation LLM, temperature 0, same max-output tokens, same system policy
 • full per-question token breakdown (system, question, context, opt-step, output, total inference)
 • BERTScore F1 (roberta-large, rescale_with_baseline=True, idf=False) vs reference answers
 • strict pass/fail + graded (0-3) LLM-judge (blind to pipeline)
 • citations (retrieved doc ids), latency
 • dev/test split, 3 runs (mean+variance), raw per-question outputs retained
 • ablation: GraphRAG with graph off / optimizer off / dedup off / mmr off

Usage:
  python -m evaluation.round3_benchmark --split test --runs 3
  python -m evaluation.round3_benchmark --ablation
"""

import os, sys, json, time, argparse, logging, statistics, subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env", override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "data" / "round3" / "round3_eval.json"
OUT = ROOT / "results" / "round3"
COST_IN, COST_OUT = 0.00015, 0.0006  # per 1k tokens (Gemini 2.5 Flash)

from backend.graph.tigergraph_client import TigerGraphClient
from backend.rag.basic_rag_tg import BasicRAGTigerGraph
from backend.rag.graph_rag_tg import GraphRAGTG
import re
from backend.llm.gemini_client import (gemini_generate, MAX_OUTPUT_TOKENS, ROUND3_SYSTEM_CORE,
                                        ROUND3_EVIDENCE_CLAUSE, count_context_tokens)

SYS_LLM = ROUND3_SYSTEM_CORE                       # LLM-only: shared core, no evidence
SYS_RAG = ROUND3_SYSTEM_CORE + ROUND3_EVIDENCE_CLAUSE   # RAG + GraphRAG (identical)
_ST_LLM = count_context_tokens(SYS_LLM)            # system+instruction tokens (estimate, tiktoken)
_ST_RAG = count_context_tokens(SYS_RAG)


# ── Pipelines return a uniform dict with FULL token breakdown ────────────────
def run_llm_only(q):
    t = time.time()
    r = gemini_generate(system_prompt=SYS_LLM, user_prompt=q, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    return {"answer": r["answer"].strip(), "prompt_tokens": r["prompt_tokens"], "output_tokens": r["completion_tokens"],
            "system_tokens": _ST_LLM, "question_tokens": count_context_tokens(q), "context_tokens": 0,
            "opt_tokens": 0, "total_inference": r["total_tokens"],
            "citations": [], "evidence": "", "latency_ms": (time.time()-t)*1000, "used_graph": False}


def run_rag(rag, q):
    r = rag.query(q)
    return {"answer": r.answer, "prompt_tokens": r.prompt_tokens, "output_tokens": r.completion_tokens,
            "system_tokens": _ST_RAG, "question_tokens": count_context_tokens(q), "context_tokens": r.context_tokens,
            "opt_tokens": 0, "total_inference": r.total_tokens,
            "citations": r.retrieved_docs, "evidence": r.evidence, "latency_ms": r.latency_ms, "used_graph": False}


def run_graphrag(gr, q, **kw):
    r = gr.query(q, **kw)
    return {"answer": r.answer, "prompt_tokens": r.prompt_tokens, "output_tokens": r.completion_tokens,
            "system_tokens": _ST_RAG, "question_tokens": count_context_tokens(q), "context_tokens": r.context_tokens,
            "opt_tokens": r.extra_llm_tokens, "total_inference": r.total_tokens + r.extra_llm_tokens,
            "citations": r.retrieved_docs, "evidence": r.evidence, "latency_ms": r.latency_ms, "used_graph": r.used_graph}


# ── Judge: strict pass/fail + graded 0-3 + evidence-support + citation — blind ─
JUDGE = """You are a strict, impartial grader for a FACTUAL filings-QA benchmark. Grade the CANDIDATE ONLY on whether it contains the specific facts and figures in the REFERENCE answer.
A fluent or plausible answer that does NOT match the reference's specific facts/figures is WRONG (grade 0-1), even if it sounds reasonable — do NOT reward unverified guesses.
Return ONLY JSON: {{"grade":<0|1|2|3>,"pass":<true|false>,"supported":<true|false>,"cite":<0|1|2>,"reason":"<short>"}}
grade 3=all key facts/figures match the reference, 2=most match, 1=partial, 0=wrong or missing the key fact. pass=true ONLY if fully correct.
supported=true if the candidate's key claims appear in the EVIDENCE shown.
cite: 2=correct source ids cited, 1=partial, 0=none/wrong.
QUESTION: {q}
REFERENCE: {ref}
EVIDENCE: {ev}
CANDIDATE: {cand}"""

def judge(q, ref, cand, evidence=""):
    try:
        raw = gemini_generate("Return only valid JSON.",
                              JUDGE.format(q=q, ref=ref, cand=cand, ev=(evidence or "(none)")[:1500]),
                              temperature=0.0, max_tokens=200)["answer"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        d = json.loads(raw)
        return int(d.get("grade", 0)), bool(d.get("pass", False)), bool(d.get("supported", False)), int(d.get("cite", 0))
    except Exception as e:
        logger.warning(f"judge fail: {e}")
        return 0, False, False, 0


def _key_nums(s):
    """Salient numeric tokens (>=3 digits) from a string, comma-stripped — for a
    deterministic, reproducible exact-figure match that the noisy LLM judge misses."""
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*", s or "") if len(n.replace(",", "")) >= 3}

def numeric_match(reference, answer):
    ref = _key_nums(reference)
    return bool(ref) and bool(ref & _key_nums(answer))  # None-like handled by caller


_FORMWORDS = {"K", "A", "DEF", "FORM", "SEC", "AND", "OR", "THE", "PART", "ITEM", "US", "DEF14A", "14A"}
def _gold_tickers(s):
    return {t for t in re.findall(r"\b[A-Z]{1,5}\b", s or "") if t not in _FORMWORDS}
def _doc_tickers(docs):
    return {d.split(":")[0] for d in (docs or [])}


def bertscore(cands, refs):
    try:
        from bert_score import score
        _, _, F1 = score(cands, refs, model_type="roberta-large", lang="en",
                         rescale_with_baseline=True, idf=False, verbose=False)
        return [round(float(x), 4) for x in F1]
    except Exception as e:
        logger.warning(f"BERTScore unavailable: {e}")
        return [None] * len(cands)


def cost(pt, ot):
    return round(pt/1000*COST_IN + ot/1000*COST_OUT, 6)


def git_info():
    """Capture the exact code version so every result is traceable to a commit."""
    def _g(args):
        try:
            return subprocess.check_output(["git"] + args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": _g(["rev-parse", "HEAD"]),
            "commit_short": _g(["rev-parse", "--short", "HEAD"]),
            "repo": _g(["config", "--get", "remote.origin.url"]),
            "dirty": bool(_g(["status", "--porcelain"]))}


def evaluate(questions, rag, gr, gr_kwargs=None, label="graphrag"):
    gr_kwargs = gr_kwargs or {}
    rows = []
    for i, item in enumerate(questions):
        q, ref = item["question"], item.get("answer_key", "")
        logger.info(f"Q{i+1}/{len(questions)} {item['id']} ({item.get('hop')}-hop)")
        res = {"id": item["id"], "hop": item.get("hop"), "tier": item.get("tier"),
               "benefit": item.get("benefit"), "source_docs": item.get("source_docs"),
               "question": q, "reference": ref}
        res["llm_only"] = run_llm_only(q)
        res["rag"] = run_rag(rag, q)
        res["graphrag"] = run_graphrag(gr, q, **gr_kwargs)
        gold = _gold_tickers(item.get("source_docs", ""))
        for p in ("llm_only", "rag", "graphrag"):
            g, pas, sup, cite = judge(q, ref, res[p]["answer"], res[p].get("evidence", ""))
            res[p]["grade"], res[p]["pass"], res[p]["supported"], res[p]["cite_score"] = g, pas, sup, cite
            got = _doc_tickers(res[p]["citations"])
            res[p]["citation_recall"] = round(len(gold & got) / len(gold), 3) if gold else None
            res[p]["citation_precision"] = round(len(gold & got) / len(got), 3) if got else None
            # evidence-quality (0-1): half from judge citation grade, half from support
            res[p]["evidence_quality"] = round(0.5 * (cite / 2) + 0.5 * (1.0 if sup else 0.0), 3)
            res[p]["numeric_match"] = numeric_match(ref, res[p]["answer"]) if _key_nums(ref) else None
            res[p]["cost"] = cost(res[p]["prompt_tokens"], res[p]["output_tokens"])
        rows.append(res)
    # BERTScore in batch per pipeline (vs reference)
    refs = [r["reference"] for r in rows]
    for p in ("llm_only", "rag", "graphrag"):
        f1 = bertscore([r[p]["answer"] for r in rows], refs)
        for r, s in zip(rows, f1):
            r[p]["bertscore_f1"] = s
    return rows


def aggregate(rows):
    n = len(rows)
    agg = {}
    for p in ("llm_only", "rag", "graphrag"):
        ti = [r[p]["total_inference"] for r in rows]
        ct = [r[p]["context_tokens"] for r in rows]
        f1 = [r[p]["bertscore_f1"] for r in rows if r[p]["bertscore_f1"] is not None]
        cr = [r[p]["citation_recall"] for r in rows if r[p].get("citation_recall") is not None]
        nm = [r[p]["numeric_match"] for r in rows if r[p].get("numeric_match") is not None]
        agg[p] = {
            "numeric_match_pct": round(sum(1 for x in nm if x)/len(nm)*100, 1) if nm else None,
            "avg_total_inference_tokens": round(sum(ti)/n, 1),
            "avg_context_tokens": round(sum(ct)/n, 1),
            "avg_system_tokens": round(sum(r[p]["system_tokens"] for r in rows)/n, 1),
            "avg_question_tokens": round(sum(r[p]["question_tokens"] for r in rows)/n, 1),
            "avg_output_tokens": round(sum(r[p]["output_tokens"] for r in rows)/n, 1),
            "strict_pass_pct": round(sum(1 for r in rows if r[p]["pass"])/n*100, 1),
            "avg_grade": round(sum(r[p]["grade"] for r in rows)/n, 2),
            "avg_bertscore_f1": round(sum(f1)/len(f1), 4) if f1 else None,
            "avg_evidence_quality": round(sum(r[p]["evidence_quality"] for r in rows)/n, 3),
            "supported_pct": round(sum(1 for r in rows if r[p]["supported"])/n*100, 1),
            "avg_citation_recall": round(sum(cr)/len(cr), 3) if cr else None,
            "avg_latency_ms": round(sum(r[p]["latency_ms"] for r in rows)/n, 1),
            "avg_cost": round(sum(r[p]["cost"] for r in rows)/n, 6),
        }
    base = agg["rag"]["avg_total_inference_tokens"] or 1
    agg["token_reduction_vs_rag_pct"] = round((base - agg["graphrag"]["avg_total_inference_tokens"])/base*100, 1)
    agg["validity_ordering_ok"] = (agg["llm_only"]["avg_grade"] <= agg["rag"]["avg_grade"] <= agg["graphrag"]["avg_grade"])

    # Per-hop breakdown — shows GraphRAG's structural advantage on multi-hop /
    # aggregation questions (the eval set is stratified by hop x fan-out).
    by_hop = {}
    for hop in sorted(set(r["hop"] for r in rows), key=str):
        grp = [r for r in rows if r["hop"] == hop]
        m = len(grp)
        rt = sum(r["rag"]["total_inference"] for r in grp) / m
        gt = sum(r["graphrag"]["total_inference"] for r in grp) / m
        by_hop[str(hop)] = {
            "n": m,
            "rag_grade": round(sum(r["rag"]["grade"] for r in grp)/m, 2),
            "graphrag_grade": round(sum(r["graphrag"]["grade"] for r in grp)/m, 2),
            "token_reduction_pct": round((rt-gt)/rt*100, 1) if rt else 0,
        }
    agg["by_hop"] = by_hop
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions (cheap smoke test)")
    ap.add_argument("--sample", type=int, default=None, help="N questions PER hop tier (cheap stratified check)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    GIT = git_info()
    logger.info(f"[GIT] repo={GIT['repo']} commit={GIT['commit_short']} dirty={GIT['dirty']}")
    qs = json.load(open(EVAL, encoding="utf-8"))
    # dev/test split: stratified, deterministic (odd index=dev, even=test) — tune on dev only
    if args.split == "dev":   qs = [q for i, q in enumerate(qs) if i % 3 == 0]
    elif args.split == "test": qs = [q for i, q in enumerate(qs) if i % 3 != 0]
    if args.sample:
        by = {}
        for q in qs:
            by.setdefault(q.get("hop"), []).append(q)
        qs = [q for grp in by.values() for q in grp[:args.sample]]
    if args.limit: qs = qs[:args.limit]
    logger.info(f"Evaluating {len(qs)} questions (split={args.split})")

    tg = TigerGraphClient().connect()
    rag, gr = BasicRAGTigerGraph(tg), GraphRAGTG(tg)

    if args.ablation:
        variants = {"full": {}, "no_graph": {"use_graph": False},
                    "no_optimizer": {"use_optimizer": False}, "no_compress": {"use_compress": False},
                    "no_dedup": {"use_dedup": False}, "no_mmr": {"use_mmr": False}}
        abl = {}
        for name, kw in variants.items():
            logger.info(f"=== ablation: {name} ===")
            rows = evaluate(qs, rag, gr, gr_kwargs=kw, label=name)
            abl[name] = aggregate(rows)["graphrag"]
        json.dump(abl, open(OUT / "ablation.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        logger.info(f"[SAVED] {OUT/'ablation.json'}")
        return

    runs = []
    for run in range(args.runs):
        logger.info(f"=== RUN {run+1}/{args.runs} ===")
        rows = evaluate(qs, rag, gr)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json.dump({"git": GIT, "aggregate": aggregate(rows), "per_question": rows},
                  open(OUT / f"run_{run+1}_{ts}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        runs.append(aggregate(rows))
        logger.info(f"[SAVED] run {run+1}")

    # One-time ingestion/indexing cost — reported SEPARATELY from per-question inference
    ingestion = {
        "embedding_model": "all-MiniLM-L6-v2 (local)",
        "embedding_api_tokens": 0,
        "note": ("Embeddings computed locally (sentence-transformers) = zero LLM/API tokens. "
                 "Graph + vector index built once in TigerGraph. All per-question numbers above "
                 "are INFERENCE tokens only."),
    }
    # mean + variance across runs
    summary = {"git": GIT, "runs": args.runs, "split": args.split, "ingestion_one_time": ingestion, "per_run": runs}
    if len(runs) > 1:
        keys = ["avg_total_inference_tokens", "strict_pass_pct", "avg_bertscore_f1", "avg_grade"]
        summary["mean_variance"] = {
            p: {k: {"mean": round(statistics.mean([r[p][k] for r in runs if r[p][k] is not None]), 3),
                    "stdev": round(statistics.pstdev([r[p][k] for r in runs if r[p][k] is not None]), 3)}
                for k in keys}
            for p in ("llm_only", "rag", "graphrag")}
    json.dump(summary, open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    logger.info(f"[DONE] {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
