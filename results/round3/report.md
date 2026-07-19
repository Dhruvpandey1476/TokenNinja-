# Round 3 — Benchmark Report

*Repo:* `https://github.com/Dhruvpandey1476/TokenNinja-.git`  ·  *Commit:* `d9dfc0e`

**Questions:** 50  |  **Token reduction (GraphRAG vs RAG):** **50.7%**  |  **Validity ordering (LLM ≤ RAG ≤ GraphRAG):** ✅ PASS

## Aggregate (per pipeline)

| Metric | LLM-Only | Traditional RAG | GraphRAG |
|---|---|---|---|
| Avg graded score (/3) | 0.64 | 1.16 | 1.38 |
| Strict pass % | 10.0 | 22.0 | 32.0 |
| Numeric-match % (exact figure) | 15.4 | 61.5 | 61.5 |
| Evidence-quality (0-1) | 0.000 | 0.535 | 0.733 |
| Citation recall | 0.000 | 0.804 | 0.902 |
| BERTScore F1 (rescaled) | -0.0142 | -0.0201 | -0.0712 |
| Avg total-inference tokens | 116 | 4141 | 2040 |
| Avg context tokens | 0 | 3496 | 1454 |
| Avg latency (ms) | 1170 | 1730 | 4558 |
| Avg cost ($) | 0.000042 | 0.000656 | 0.000357 |


## By hop tier (GraphRAG's structural advantage)

| Hop | n | RAG grade | GraphRAG grade | Token ↓ |
|---|---|---|---|---|
| 1 | 9 | 2.22 | 2.56 | 52.0% |
| 2 | 20 | 1.25 | 1.35 | 51.5% |
| 3+ | 21 | 0.62 | 0.9 | 49.3% |


## Ablation (GraphRAG components)

| Variant | Grade | Pass % | Tokens | Evidence-Q |
|---|---|---|---|---|
| full | 1.42 | 34.0 | 2035 | 0.743 |
| no_graph | 1.06 | 22.0 | 1811 | 0.735 |
| no_optimizer | 0.78 | 12.0 | 2039 | 0.705 |
| no_compress | 1.44 | 32.0 | 4314 | 0.743 |
| no_dedup | 1.40 | 34.0 | 2031 | 0.743 |
| no_mmr | 1.12 | 24.0 | 2011 | 0.682 |


## Reproducibility (mean ± stdev across runs)

| Pipeline | Grade (mean±sd) | Tokens (mean±sd) |
|---|---|---|
| LLM-Only | 0.68 ± 0.033 | 126.667 ± 7.403 |
| Traditional RAG | 1.153 ± 0.025 | 4137.433 ± 3.19 |
| GraphRAG | 1.38 ± 0.0 | 2034.0 ± 4.744 |


## One-time ingestion cost (separate from inference)
- Embedding API tokens: **0** (all-MiniLM-L6-v2 (local))
- Embeddings computed locally (sentence-transformers) = zero LLM/API tokens. Graph + vector index built once in TigerGraph. All per-question numbers above are INFERENCE tokens only.


## Takeaways

- GraphRAG uses **50.7% fewer inference tokens** than Traditional RAG.

- GraphRAG graded **1.38/3** vs RAG **1.16/3**; strict pass **32.0%** vs **22.0%**.

- Evidence quality **0.733** vs **0.535** — GraphRAG citations are more precise.

- Numeric-match shows retrieval is essential: LLM-Only **15.4%** vs GraphRAG **61.5%**.
