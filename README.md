# 🐯 TokenNinja — Round 3: Generalizable Context Optimization

**TigerGraph GraphRAG Hackathon — Round 3.** Three pipelines compared head-to-head on the shared SP100 SEC-filings dataset, proving that graph-based retrieval + context optimization cuts **total inference tokens** while preserving answer quality.

## Pipelines
1. **LLM-Only** — question → Gemini, no retrieval (baseline floor).
2. **Traditional RAG** — TigerGraph **vector** search (top-k) → Gemini.
3. **Optimized GraphRAG** — TigerGraph vector search **+ graph traversal (2-hop fusion)** → deterministic context optimizer (dedup + MMR + graph-aware scoring + sentence compression) → Gemini.

TigerGraph Savanna is used as **both the graph and vector database**. RAG and GraphRAG share the **same embedding model** (`all-MiniLM-L6-v2`) and the **same generation config** (Gemini 2.5 Flash, temperature 0, shared system prompt) — only retrieval + optimization differ.

## Dataset
`data/sp100_dataset/` — 100 S&P companies × {10-K, 8-K, DEF14A} SEC filings (heterogeneous: financial facts, governance, events). Eval set: `data/round3/round3_eval.json` (50 questions stratified by hop depth × fan-out).

## Results (50 questions · 3 runs · commit-traceable)

| Metric | LLM-Only | Traditional RAG | **GraphRAG** |
|---|---|---|---|
| Graded score (/3) | 0.64 | 1.16 | **1.38** |
| Strict pass rate | 10% | 22% | **32%** |
| Numeric-match (exact figure) | 15% | 62% | **62%** |
| Evidence-quality (0–1) | 0.00 | 0.54 | **0.73** |
| Avg total-inference tokens | 116 | 4,141 | **2,040** |
| **Token reduction vs RAG** | — | — | **−50.7%** |

- **Validity ordering holds:** LLM-Only (0.64) **< RAG (1.16) < GraphRAG (1.38)** — retrieval provably helps, and GraphRAG is best.
- **Wins every hop tier** (graded): 1-hop 2.56 vs 2.22 · 2-hop 1.35 vs 1.25 · 3-hop+ 0.90 vs 0.62 — GraphRAG's edge grows with hop depth.
- **Reproducible:** 3 independent runs, GraphRAG grade stdev ≈ 0.
- **Retrieval is essential:** LLM-Only answers only 15% of figures correctly vs 62% for the retrieval pipelines.

### Ablation (contribution of each GraphRAG component)
| Variant | Grade /3 | Tokens | Takeaway |
|---|---|---|---|
| **Full** | 1.42 | 2,035 | — |
| No graph | 1.06 | 1,811 | graph traversal adds **+0.36 grade** |
| No optimizer | 0.78 | 2,039 | optimizer is critical |
| No compression | 1.44 | 4,315 | **compression halves tokens with ~no accuracy loss** |
| No MMR | 1.12 | 2,011 | diversity ranking adds +0.30 |

### Fairness & rigor
Same LLM (Gemini 2.5 Flash, temp 0), same output cap, same embedding model, and one shared core system prompt across all pipelines. The judge grades **evidence-blind** against the reference answer — identical for every pipeline. Support and citations are scored **deterministically**. Every result carries the git commit it was produced from.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # fill TigerGraph + Gemini credentials
```

## Run
```bash
# 1. Build chunks + graph backbone from the corpus ($0, offline)
python -m scripts.round3_ingest

# 2. Load schema + graph + vectors into TigerGraph Savanna
python -m scripts.round3_load_tigergraph

# 3. Benchmark (tune on dev, then held-out test, 3 runs)
python -m evaluation.round3_benchmark --split dev
python -m evaluation.round3_benchmark --split all --runs 3

# 4. Ablation (contribution of graph / optimizer / compression)
python -m evaluation.round3_benchmark --ablation
```

## What the result files contain (per question × pipeline)
Answer · retrieved doc/chunk ids (citations) · evidence supplied · token breakdown (system / question / context / output / total-inference) · BERTScore F1 (`roberta-large`, rescaled) · strict pass + graded LLM-judge · evidence/citation-quality · deterministic numeric-match · latency. Plus per-hop breakdown, aggregate, ingestion cost (reported separately), and 3-run mean/variance.

## Key design choices
- **Context optimizer is deterministic** → zero extra inference tokens.
- **Adaptive budget** → single-hop stays tight; high-fan-out aggregation gets room for one fact per company.
- **2-hop fusion** → bridge questions (e.g. "the company that acquired X → its auditor") retrieve the second entity's filings.
- **Fair comparison** → same model, temperature, output cap, and core system instruction across all pipelines; every pipeline-specific difference is disclosed and ablated.

## Deliverables
- **Evaluation dashboard:** [`docs/dashboard.html`](docs/dashboard.html) — 3-pipeline comparison, per-hop, ablation (open in a browser)
- **Live 3-pipeline app:** `python -m scripts.query "your question"` — runs LLM-Only vs RAG vs GraphRAG side-by-side
- **Architecture & tech stack:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Methodology presentation:** [`docs/presentation.html`](docs/presentation.html) (arrow keys / click to navigate)
- **Blog / social post:** [`docs/BLOG.md`](docs/BLOG.md)
- **Raw results:** `results/round3/` — `run_1..3.json`, `summary.json`, `ablation.json`, `report.md`, `report.csv`

*Built for the TigerGraph GraphRAG Hackathon. #GraphRAGInferenceHackathon*
