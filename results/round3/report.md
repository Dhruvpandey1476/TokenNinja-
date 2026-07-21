# Round 3 — Benchmark Report

*Repo:* `https://github.com/Dhruvpandey1476/TokenNinja-.git`  ·  *Commit:* `5af4540`

**Questions:** 50  |  **Token reduction (GraphRAG vs RAG):** **51.0%**  |  **Validity ordering (LLM ≤ RAG ≤ GraphRAG):** ✅ PASS

## Aggregate (per pipeline)

| Metric | LLM-Only | Traditional RAG | GraphRAG |
|---|---|---|---|
| Avg graded score (/3) | 0.72 | 1.26 | 1.48 |
| Strict pass % | 10.0 | 26.0 | 34.0 |
| Numeric-match % (exact figure) | 15.4 | 61.5 | 61.5 |
| Evidence-quality (0-1) | 0.000 | 0.545 | 0.743 |
| Citation recall | 0.000 | 0.804 | 0.902 |
| BERTScore F1 (rescaled) | -0.0228 | -0.0101 | -0.0597 |
| Avg total-inference tokens | 132 | 4137 | 2028 |
| Avg context tokens | 0 | 3496 | 1449 |
| Avg latency (ms) | 1346 | 1643 | 2412 |
| Avg cost ($) | 0.000051 | 0.000654 | 0.000353 |


## By hop tier (GraphRAG's structural advantage)

| Hop | n | RAG grade | GraphRAG grade | Token ↓ |
|---|---|---|---|---|
| 1 | 9 | 2.56 | 2.56 | 52.0% |
| 2 | 20 | 1.25 | 1.55 | 51.6% |
| 3+ | 21 | 0.71 | 0.95 | 49.8% |


## Ablation (GraphRAG components)

| Variant | Grade | Pass % | Tokens | Evidence-Q |
|---|---|---|---|---|
| full | 1.46 | 32.0 | 2031 | 0.733 |
| no_graph | 1.00 | 20.0 | 1811 | 0.735 |
| no_optimizer | 0.82 | 12.0 | 1982 | 0.693 |
| no_compress | 1.42 | 30.0 | 4318 | 0.752 |
| no_dedup | 1.50 | 34.0 | 2030 | 0.733 |
| no_mmr | 1.12 | 24.0 | 2018 | 0.682 |


## Reproducibility (mean ± stdev across runs)

| Pipeline | Grade (mean±sd) | Tokens (mean±sd) |
|---|---|---|
| LLM-Only | 0.707 ± 0.009 | 124.7 ± 5.838 |
| Traditional RAG | 1.26 ± 0.016 | 4136.733 ± 0.45 |
| GraphRAG | 1.467 ± 0.009 | 2030.233 ± 1.721 |


## One-time ingestion cost (separate from inference)
- Embedding API tokens: **0** (all-MiniLM-L6-v2 (local))
- Embeddings computed locally (sentence-transformers) = zero LLM/API tokens. Graph + vector index built once in TigerGraph. Chunk embeddings are also precomputed once at index time (scripts/build_embedding_cache.py) and reused at inference — a standard index-time optimization, 0 API tokens, identical vectors. All per-question numbers above are INFERENCE tokens only.


## Takeaways

- GraphRAG uses **51.0% fewer inference tokens** than Traditional RAG.

- GraphRAG graded **1.48/3** vs RAG **1.26/3**; strict pass **34.0%** vs **26.0%**.

- Evidence quality **0.743** vs **0.545** — GraphRAG citations are more precise.

- Numeric-match shows retrieval is essential: LLM-Only **15.4%** vs GraphRAG **61.5%**.
