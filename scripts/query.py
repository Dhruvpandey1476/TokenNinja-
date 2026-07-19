#!/usr/bin/env python3
"""
Three-pipeline application — run ONE question through LLM-Only, Traditional RAG,
and Optimized GraphRAG live, and print a side-by-side comparison (answer, tokens,
citations, latency). Lets judges interactively compare the pipelines.

Usage:
  python -m scripts.query "Apple's total net sales for fiscal year 2025?"
  python -m scripts.query                      # interactive prompt loop
"""

import sys, time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from backend.graph.tigergraph_client import TigerGraphClient
from backend.rag.basic_rag_tg import BasicRAGTigerGraph
from backend.rag.graph_rag_tg import GraphRAGTG
from backend.llm.gemini_client import gemini_generate, MAX_OUTPUT_TOKENS, ROUND3_SYSTEM_CORE


def run(q, rag, gr):
    t = time.time()
    r1 = gemini_generate(ROUND3_SYSTEM_CORE, q, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    llm = {"name": "LLM-Only", "answer": r1["answer"].strip(), "tokens": r1["total_tokens"],
           "cites": [], "ms": (time.time()-t)*1000}
    r2 = rag.query(q)
    rr = {"name": "Traditional RAG", "answer": r2.answer, "tokens": r2.total_tokens,
          "cites": r2.retrieved_docs, "ms": r2.latency_ms}
    r3 = gr.query(q)
    gg = {"name": "GraphRAG", "answer": r3.answer, "tokens": r3.total_tokens,
          "cites": r3.retrieved_docs, "ms": r3.latency_ms}
    return [llm, rr, gg]


def show(q, results):
    print("\n" + "=" * 78)
    print("Q:", q)
    print("=" * 78)
    base = next((r["tokens"] for r in results if r["name"] == "Traditional RAG"), 0)
    for r in results:
        red = f"  ({(base-r['tokens'])/base*100:+.0f}% vs RAG)" if base and r["name"] == "GraphRAG" else ""
        print(f"\n### {r['name']}  |  tokens={r['tokens']}{red}  |  latency={r['ms']:.0f}ms")
        print(r["answer"])
        if r["cites"]:
            print("citations:", ", ".join(r["cites"][:6]))


def main():
    tg = TigerGraphClient().connect()
    rag, gr = BasicRAGTigerGraph(tg), GraphRAGTG(tg)
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        show(q, run(q, rag, gr))
    else:
        print("Three-pipeline comparison. Type a question (blank to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            show(q, run(q, rag, gr))


if __name__ == "__main__":
    main()
