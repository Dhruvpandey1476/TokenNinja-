# LinkedIn Post

---

🐯 Just wrapped **Round 3 of the TigerGraph GraphRAG Hackathon** — the "Generalizable Context Optimization" challenge — and I'm genuinely proud of this one.

**The problem:** GraphRAG retrieves through vectors *and* graph relationships, so it finds more evidence than plain RAG — but it also hands the LLM a lot of duplicate, overlapping context. More tokens, more cost, no guaranteed accuracy gain. This round, every team worked on the **same** heterogeneous dataset — 100 S&P-100 companies' SEC filings (10-Ks, 8-Ks, proxy statements) — so there was nowhere to hide.

**What I built:** a pipeline that retrieves broadly, then optimizes the evidence **deterministically** — semantic dedup (keeping *complementary* facts, not just removing duplicates), MMR diversity ranking, graph-aware scoring, and adaptive sentence-level compression. Because it's deterministic, it adds **zero extra inference tokens**. TigerGraph Savanna does double duty as **both the graph and the vector database**, so vector search *and* multi-hop traversal happen in-database.

**The results (50 questions, 3 runs, stdev ≈ 0):**
✅ **−51% total inference tokens** vs traditional RAG
✅ **Higher** accuracy, not lower — graded 1.47/3 vs RAG's 1.26, strict pass 34% vs 25%
✅ Valid benchmark ordering: LLM-only (0.71) < RAG (1.26) < GraphRAG (1.47)
✅ GraphRAG wins **every** hop tier — and its edge grows with hop depth (the graph earns its keep on multi-hop reasoning)
✅ And it's **balanced**, not token-obsessed: ~2.4s latency vs RAG's ~1.7s (embeddings precomputed at index time, so the optimizer adds ~0 query-time cost)

**My favorite part — the ablation:**
🔹 Remove the graph → accuracy drops 1.46 → 1.00 (the graph adds real accuracy)
🔹 Remove compression → tokens *double* (2.0k → 4.3k) with ~no accuracy change (the savings are free)

A few lessons that surprised me:
• **Deterministic beats LLM-based** for context optimization — reproducible, and it doesn't spend the tokens you're trying to save.
• **Grade answers evidence-blind** — showing a judge each pipeline's (variable-length) context quietly biases the scores.
• **Two-hop "bridge" retrieval** — e.g. "the company that acquired X → *that* company's auditor" — is where graphs do something vectors simply can't.

Huge thanks to the TigerGraph team for a genuinely rigorous challenge — it pushed me to care about *provable, reproducible* wins over headline numbers.

Code + full results + ablation 👉 github.com/Dhruvpandey1476/TokenNinja-

#GraphRAGInferenceHackathon #TigerGraph #GraphRAG #RAG #LLM #AI #KnowledgeGraphs

---

