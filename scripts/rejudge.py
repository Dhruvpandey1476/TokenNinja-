#!/usr/bin/env python3
"""
Re-judge saved benchmark answers with an EVIDENCE-BLIND, reference-only grader.

Why: the original judge saw each pipeline's retrieved evidence, and long contexts
(Traditional RAG) were truncated in the judge prompt — unfairly lowering their
grade. Grading correctness against the reference answer alone, identically for
every pipeline, removes that bias. This is a grading-methodology fix ONLY:
  • no pipeline answer is changed (answers are read from the saved run)
  • the SAME judge is applied to all three pipelines
  • the original run file is left intact; output is a new, clearly-labelled file
Cost: only the judge calls (~3 x #questions), no retrieval/generation re-run.

Usage:
  python -m scripts.rejudge                    # latest run_*.json
  python -m scripts.rejudge --file <path>
"""

import json, glob, os, argparse
from pathlib import Path
from backend.llm.gemini_client import gemini_generate

OUT = Path(__file__).parent.parent / "results" / "round3"
PIPES = ["llm_only", "rag", "graphrag"]

JUDGE = """You are a strict, impartial grader for a FACTUAL filings-QA benchmark. Grade the CANDIDATE ONLY on whether it contains the specific facts/figures in the REFERENCE answer.
A fluent or plausible answer that does NOT match the reference's facts/figures is WRONG (grade 0-1) — do not reward unverified guesses.
You grade correctness against the REFERENCE only; you are blind to which pipeline produced the answer and to any retrieved context, so every answer is judged identically.
Return ONLY JSON: {{"grade":<0|1|2|3>,"pass":<true|false>,"reason":"<short>"}}
grade 3=all key facts/figures match, 2=most, 1=partial, 0=wrong/missing. pass=true ONLY if fully correct.
QUESTION: {q}
REFERENCE: {ref}
CANDIDATE: {cand}"""


def judge(q, ref, cand):
    try:
        raw = gemini_generate("Return only valid JSON.", JUDGE.format(q=q, ref=ref, cand=cand),
                              temperature=0.0, max_tokens=150)["answer"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        d = json.loads(raw)
        return int(d.get("grade", 0)), bool(d.get("pass", False))
    except Exception as e:
        print("judge fail:", e)
        return 0, False


def reaggregate(rows, orig_agg):
    """Recompute grade/pass/ordering from re-judged rows; keep all other agg fields."""
    n = len(rows)
    agg = json.loads(json.dumps(orig_agg))  # deep copy, preserves tokens/numeric/etc.
    for p in PIPES:
        agg[p]["avg_grade"] = round(sum(r[p]["grade"] for r in rows) / n, 2)
        agg[p]["strict_pass_pct"] = round(sum(1 for r in rows if r[p]["pass"]) / n * 100, 1)
    agg["validity_ordering_ok"] = (agg["llm_only"]["avg_grade"] <= agg["rag"]["avg_grade"]
                                   <= agg["graphrag"]["avg_grade"])
    for hop in agg.get("by_hop", {}):
        grp = [r for r in rows if str(r.get("hop")) == hop]
        if grp:
            agg["by_hop"][hop]["rag_grade"] = round(sum(r["rag"]["grade"] for r in grp) / len(grp), 2)
            agg["by_hop"][hop]["graphrag_grade"] = round(sum(r["graphrag"]["grade"] for r in grp) / len(grp), 2)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    f = args.file or sorted(glob.glob(str(OUT / "run_*.json")), key=os.path.getmtime)[-1]
    d = json.load(open(f, encoding="utf-8"))
    rows, orig = d["per_question"], d["aggregate"]
    print(f"Re-judging {len(rows)} questions x 3 pipelines (evidence-blind)...")

    for i, r in enumerate(rows):
        q, ref = r["question"], r["reference"]
        for p in PIPES:
            g, pas = judge(q, ref, r[p]["answer"])
            r[p]["grade"], r[p]["pass"] = g, pas
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}")

    new_agg = reaggregate(rows, orig)
    out = {"git": d.get("git"), "grading": "evidence-blind reference-only re-judge (methodology fix, applied identically to all pipelines; answers unchanged)",
           "aggregate": new_agg, "per_question": rows}
    path = OUT / (os.path.basename(f).replace("run_", "rejudged_"))
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== BEFORE (original judge) ===")
    for p in PIPES:
        print(f"  {p:9} grade={orig[p]['avg_grade']:.2f} pass={orig[p]['strict_pass_pct']:.0f}%")
    print("  ordering_ok:", orig.get("validity_ordering_ok"))
    print("=== AFTER (evidence-blind) ===")
    for p in PIPES:
        print(f"  {p:9} grade={new_agg[p]['avg_grade']:.2f} pass={new_agg[p]['strict_pass_pct']:.0f}%")
    print("  ordering_ok:", new_agg.get("validity_ordering_ok"))
    print(f"\n[SAVED] {path}")


if __name__ == "__main__":
    main()
