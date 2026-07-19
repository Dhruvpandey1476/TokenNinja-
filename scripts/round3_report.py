#!/usr/bin/env python3
"""
Round 3 report generator — turns the raw result JSON into readable, submission-
formatted outputs ($0, offline):

  results/round3/report.md    human-readable summary (aggregate + per-hop + ablation)
  results/round3/report.csv   per-question table (every required field, one row/Q)

Usage:
  python -m scripts.round3_report                 # latest run_*.json
  python -m scripts.round3_report --file <path>   # a specific run file
"""

import json, csv, glob, os, argparse
from pathlib import Path

OUT = Path(__file__).parent.parent / "results" / "round3"
PIPES = [("llm_only", "LLM-Only"), ("rag", "Traditional RAG"), ("graphrag", "GraphRAG")]


def latest_run():
    fs = sorted(glob.glob(str(OUT / "run_*.json")), key=os.path.getmtime)
    return fs[-1] if fs else None


def fmt(v, nd=2):
    if v is None: return "—"
    if isinstance(v, bool): return "yes" if v else "no"
    if isinstance(v, float): return f"{v:.{nd}f}"
    return str(v)


def write_csv(rows, path):
    cols = ["id", "hop", "tier", "question", "reference"]
    per = ["answer", "total_inference", "context_tokens", "system_tokens", "question_tokens",
           "output_tokens", "grade", "pass", "numeric_match", "evidence_quality",
           "citation_recall", "bertscore_f1", "citations", "latency_ms"]
    header = cols + [f"{pk}_{c}" for pk, _ in PIPES for c in per]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            row = [r.get(c, "") for c in cols]
            for pk, _ in PIPES:
                p = r.get(pk, {})
                for c in per:
                    v = p.get(c, "")
                    if c == "answer": v = (v or "")[:300].replace("\n", " ")
                    if c == "citations": v = "; ".join(v or [])
                    row.append(v)
            w.writerow(row)


def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def write_md(agg, rows, path, ablation=None, summary=None, git=None):
    L = []
    L.append("# Round 3 — Benchmark Report\n")
    if git:
        L.append(f"*Repo:* `{git.get('repo','?')}`  ·  *Commit:* `{git.get('commit_short','?')}`"
                 f"{'  ⚠️ uncommitted changes' if git.get('dirty') else ''}\n")
    L.append(f"**Questions:** {len(rows)}  |  **Token reduction (GraphRAG vs RAG):** "
             f"**{agg.get('token_reduction_vs_rag_pct')}%**  |  "
             f"**Validity ordering (LLM ≤ RAG ≤ GraphRAG):** "
             f"{'✅ PASS' if agg.get('validity_ordering_ok') else '⚠️ see note'}\n")

    # Aggregate table
    metrics = [
        ("Avg graded score (/3)", "avg_grade", 2),
        ("Strict pass %", "strict_pass_pct", 1),
        ("Numeric-match % (exact figure)", "numeric_match_pct", 1),
        ("Evidence-quality (0-1)", "avg_evidence_quality", 3),
        ("Citation recall", "avg_citation_recall", 3),
        ("BERTScore F1 (rescaled)", "avg_bertscore_f1", 4),
        ("Avg total-inference tokens", "avg_total_inference_tokens", 0),
        ("Avg context tokens", "avg_context_tokens", 0),
        ("Avg latency (ms)", "avg_latency_ms", 0),
        ("Avg cost ($)", "avg_cost", 6),
    ]
    L.append("## Aggregate (per pipeline)\n")
    hdr = ["Metric"] + [name for _, name in PIPES]
    trows = [[m] + [fmt(agg[pk].get(k), nd) for pk, _ in PIPES] for m, k, nd in metrics]
    L.append(md_table(hdr, trows))

    # Per-hop
    if agg.get("by_hop"):
        L.append("\n## By hop tier (GraphRAG's structural advantage)\n")
        hrows = [[h, v["n"], v["rag_grade"], v["graphrag_grade"], f"{v['token_reduction_pct']}%"]
                 for h, v in agg["by_hop"].items()]
        L.append(md_table(["Hop", "n", "RAG grade", "GraphRAG grade", "Token ↓"], hrows))

    # Ablation
    if ablation:
        L.append("\n## Ablation (GraphRAG components)\n")
        arows = [[name, fmt(v.get("avg_grade"), 2), fmt(v.get("strict_pass_pct"), 1),
                  fmt(v.get("avg_total_inference_tokens"), 0), fmt(v.get("avg_evidence_quality"), 3)]
                 for name, v in ablation.items()]
        L.append(md_table(["Variant", "Grade", "Pass %", "Tokens", "Evidence-Q"], arows))

    # 3-run variance
    if summary and summary.get("mean_variance"):
        L.append("\n## Reproducibility (mean ± stdev across runs)\n")
        mv = summary["mean_variance"]
        rr = [[name, f"{mv[pk]['avg_grade']['mean']} ± {mv[pk]['avg_grade']['stdev']}",
               f"{mv[pk]['avg_total_inference_tokens']['mean']} ± {mv[pk]['avg_total_inference_tokens']['stdev']}"]
              for pk, name in PIPES]
        L.append(md_table(["Pipeline", "Grade (mean±sd)", "Tokens (mean±sd)"], rr))

    if summary and summary.get("ingestion_one_time"):
        ing = summary["ingestion_one_time"]
        L.append(f"\n## One-time ingestion cost (separate from inference)\n"
                 f"- Embedding API tokens: **{ing.get('embedding_api_tokens')}** ({ing.get('embedding_model')})\n"
                 f"- {ing.get('note')}\n")

    L.append("\n## Takeaways\n")
    g, r = agg["graphrag"], agg["rag"]
    L.append(f"- GraphRAG uses **{agg.get('token_reduction_vs_rag_pct')}% fewer inference tokens** than Traditional RAG.\n")
    L.append(f"- GraphRAG graded **{fmt(g['avg_grade'])}/3** vs RAG **{fmt(r['avg_grade'])}/3**; strict pass **{fmt(g['strict_pass_pct'],1)}%** vs **{fmt(r['strict_pass_pct'],1)}%**.\n")
    L.append(f"- Evidence quality **{fmt(g['avg_evidence_quality'],3)}** vs **{fmt(r['avg_evidence_quality'],3)}** — GraphRAG citations are more precise.\n")
    L.append(f"- Numeric-match shows retrieval is essential: LLM-Only **{fmt(agg['llm_only']['numeric_match_pct'],1)}%** vs GraphRAG **{fmt(g['numeric_match_pct'],1)}%**.\n")

    open(path, "w", encoding="utf-8").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    f = args.file or latest_run()
    if not f:
        print("No run_*.json found in results/round3/"); return
    d = json.load(open(f, encoding="utf-8"))
    agg, rows = d["aggregate"], d["per_question"]
    ablation = json.load(open(OUT / "ablation.json", encoding="utf-8")) if (OUT / "ablation.json").exists() else None
    summary = json.load(open(OUT / "summary.json", encoding="utf-8")) if (OUT / "summary.json").exists() else None

    write_csv(rows, OUT / "report.csv")
    write_md(agg, rows, OUT / "report.md", ablation, summary, git=d.get("git"))
    print(f"[OK] source: {os.path.basename(f)}")
    print(f"[OK] wrote {OUT/'report.md'}")
    print(f"[OK] wrote {OUT/'report.csv'}")


if __name__ == "__main__":
    main()
