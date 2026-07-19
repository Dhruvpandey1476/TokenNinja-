# Cutting GraphRAG's Token Bill in Half — Without Losing Accuracy

### How we built a context-optimization pipeline on TigerGraph that beats vector-RAG on cost *and* quality across a heterogeneous enterprise dataset

*TigerGraph GraphRAG Hackathon — Round 3: Generalizable Context Optimization*

![Architecture](ARCHITECTURE_DIAGRAM.svg)

---

## 1. The problem: GraphRAG's recall is a double-edged sword

GraphRAG retrieves evidence through *both* vector similarity *and* graph relationships — entities, neighborhoods, multi-hop connections. That higher recall surfaces evidence traditional RAG misses. But it comes at a price: the model often receives **far more context than it needs** — duplicate passages, overlapping facts, and marginally relevant chunks. More context means **more tokens, higher cost, and higher latency — with no guaranteed accuracy gain.**

Round 3's challenge was precise: **transform high-recall GraphRAG retrieval into a smaller, more precise, higher-quality evidence set — *before* generation** — and prove it *generalizes*. Every team worked on the **same** dataset, question set, and evaluation harness: 100 S&P-100 companies' SEC filings (10-Ks, 8-Ks, proxy statements) — a genuinely heterogeneous mix of structured financial facts, long-form governance narrative, and event disclosures. No dataset-specific tricks allowed.

And there's a subtlety the challenge called out explicitly: **this is not just deduplication.** Two near-identical passages may contain *different* numbers, dates, exceptions, or relationships. A good optimizer must tell **repeated** information apart from **complementary** evidence.

## 2. The setup: three pipelines, one controlled experiment

We implemented and compared three pipelines under **identical generation conditions**:

1. **LLM-Only** — question → Gemini 2.5 Flash, no retrieval. The floor.
2. **Traditional RAG** — TigerGraph vector search (top-k = 8) → Gemini. A competent, good-faith baseline.
3. **Optimized GraphRAG** — TigerGraph vector search **+ 2-hop graph fusion → context optimizer** → Gemini.

Crucially, **TigerGraph Savanna is used as both the graph and the vector database** — chunk embeddings live in a native vector attribute (HNSW, cosine) alongside the Company/Filing/Sector graph, so vector search *and* multi-hop traversal happen in-database. RAG and GraphRAG share the **same embedding model** (`all-MiniLM-L6-v2`), and all three share the **same model, temperature (0), output cap, and core system prompt.** Only retrieval and optimization differ — exactly the variables under test.

## 3. The core idea: retrieve broadly, then optimize *deterministically*

The heart of the system is a context optimizer that sits between retrieval and generation. Every stage is **deterministic** — which means it adds **zero inference tokens**. That matters enormously, because Round 3 scores *total inference tokens across all model calls*: an LLM-based reranker or summarizer would spend the very tokens we're trying to save. Ours doesn't.

**Step 1 — Vector seed.** A high-recall kNN query over chunk embeddings in TigerGraph pulls candidate evidence.

**Step 2 — 2-hop graph fusion.** This is where the graph earns its keep. We expand not only on entities named in the question, but on **bridge entities discovered in the first-hop evidence.** Example: *"Who is the independent auditor of the company that agreed to acquire Apogee Therapeutics?"* — the first hop finds AbbVie (in an 8-K); the second hop then retrieves **AbbVie's proxy statement** to surface the auditor. The second entity is *discovered*, not given. Company→Filing→Chunk and Company→Sector edges make sector-wide aggregation ("which Information-Technology companies…") a graph operation, not a brute-force scan.

**Step 3 — Optimize.** On the fused candidate set we run:
- **Semantic dedup** that keeps *complementary* passages — a near-duplicate is retained if it introduces a new figure, date, or entity.
- **MMR (Maximal Marginal Relevance)** for diversity — coverage of the answer, not five copies of one fact.
- **Graph-aware boosting** — evidence tied to the query's graph entities ranks higher.
- **Sentence-level compression to an *adaptive budget*** — tight (~1 fact) for single-hop lookups, larger for many-company aggregation. Answer-bearing sentences (those carrying figures + query terms) are explicitly preserved so exact numbers survive compression.

**Step 4 — Generate** with the compact evidence, ending each answer with a `Sources: [ids]` citation line.

## 4. Results (50 questions, 3 runs)

| Metric | LLM-Only | Traditional RAG | **GraphRAG** |
|---|---|---|---|
| Graded score (/3) | 0.64 | 1.16 | **1.38** |
| Strict pass rate | 10% | 22% | **32%** |
| Exact-figure match (deterministic) | 15% | 62% | **62%** |
| Evidence-quality (0–1) | 0.00 | 0.54 | **0.73** |
| Avg total-inference tokens | 116 | 4,141 | **2,040** |
| **Token reduction vs RAG** | — | — | **−50.7%** |

- **The benchmark is valid:** LLM-Only (0.64) **< RAG (1.16) < GraphRAG (1.38).** Retrieval provably helps, the baseline is sound, and GraphRAG is best — the ordering that separates a real result from an unverifiable one.
- **Retrieval is clearly necessary:** on a synthetic/future-dated corpus, the LLM answers only **15%** of figures correctly from parametric memory; both retrieval pipelines hit **62%.**
- **GraphRAG wins every hop tier, and its edge grows with hop depth:** 1-hop 2.56 vs 2.22 · 2-hop 1.35 vs 1.25 · **3-hop+ 0.90 vs 0.62.** Exactly where cross-document reasoning matters most.

## 5. Ablation: proving *why* it works

A headline number means little without showing which mechanism produced it. We ran GraphRAG with each component disabled:

| Variant | Grade /3 | Tokens | What it proves |
|---|---|---|---|
| **Full** | 1.42 | 2,035 | — |
| − graph traversal | 1.06 | 1,811 | **the graph contributes +0.36 grade** |
| − optimizer | 0.78 | 2,039 | the optimizer is critical to accuracy |
| − compression | 1.44 | 4,315 | **compression halves tokens with ~no accuracy loss** |
| − MMR | 1.12 | 2,011 | diversity ranking adds +0.30 |

The two rows that matter most: removing the **graph** costs real accuracy (1.42 → 1.06), and removing **compression** roughly **doubles tokens (2,035 → 4,315) for essentially the same grade.** That's the whole thesis, quantified: *the graph adds accuracy; the compression makes it cheap — and the savings are free.*

## 6. What we learned

- **Deterministic optimization beats LLM-based filtering here.** It's reproducible and, critically, costs zero of the tokens you're accounting for.
- **Grade evidence-blind.** Our first judge saw each pipeline's retrieved context — and because RAG's context is large, it got truncated in the judge prompt and was unfairly marked "unsupported." Grading answers against the *reference only*, identically for every pipeline, is both fairer and more reproducible. (We also added a deterministic exact-figure match as an objective cross-check the LLM judge can't wobble on.)
- **Adaptive budgets are essential on heterogeneous data.** A single-hop financial lookup needs one fact; a sector-wide aggregation needs one fact per company. A fixed budget fails one or the other.
- **Two-hop fusion is what makes "bridge" questions solvable** — and it's a genuinely graph-native operation, not something a vector index can do.

## 7. Rigor & reproducibility

Same model/params across pipelines; dev/test discipline; the held-out set run **3×** (GraphRAG grade stdev ≈ 0); full per-question token accounting (system / question / context / output / total-inference); one-time ingestion cost reported separately (embeddings are computed locally → **0 API tokens**); and **every result stamped with the git commit that produced it.** TigerGraph Savanna serves both graph and vector retrieval. Everything is in the public repo.

## 8. Conclusion

On a shared, heterogeneous enterprise dataset, a graph-native retrieval pipeline with deterministic context optimization delivered **the same-or-better accuracy as vector-RAG at half the inference tokens**, with the graph's contribution and the compression's cost-freeness both proven by ablation. The win isn't a bigger number — it's a *rigorously demonstrated, reproducible* one.

*Code & full results: [github.com/Dhruvpandey1476/TokenNinja-](https://github.com/Dhruvpandey1476/TokenNinja-) · Built with TigerGraph Savanna + Gemini 2.5 Flash · #GraphRAGInferenceHackathon*
