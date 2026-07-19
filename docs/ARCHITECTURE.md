# Architecture & Technology Stack — TokenNinja (Round 3)

## Overview
Three pipelines answer the same 50 questions on the same SP100 SEC-filings corpus, under identical generation conditions. Only **retrieval + context optimization** differ — those are the experimental variables.

```
                         ┌───────────────────────── same corpus (SP100 SEC filings) ─────────────────────────┐
                         │                                                                                    │
  Question ──► LLM-Only ─┼─► Gemini 2.5 Flash ──────────────────────────────────────────────► Answer          │
                         │                                                                                    │
             Traditional ├─► TigerGraph VECTOR search (top-k=8) ─────────────► Gemini ───────► Answer + cites  │
                    RAG   │                                                                                    │
             Optimized   └─► TigerGraph VECTOR search (recall) ─► GRAPH 2-hop fusion ─► Context Optimizer ─►    │
             GraphRAG        (seed chunks)          (Company/Filing/Sector traversal)   (dedup+MMR+graph+       │
                                                                                         sentence compression)  │
                                                                              ─► Gemini ─► Answer + cites       │
```

## Technology stack
| Layer | Choice |
|---|---|
| Language model (generation + judge) | **Gemini 2.5 Flash**, temperature 0, shared max-output cap |
| Embedding model (RAG **and** GraphRAG) | **sentence-transformers `all-MiniLM-L6-v2`** (384-dim) |
| Graph **and** vector database | **TigerGraph Savanna** — native vector attribute (`Chunk.emb`, HNSW, cosine) + graph traversal |
| Backend | Python, `pyTigerGraph`, `google-genai` |
| Evaluation | LLM-as-Judge (graded 0–3 + strict pass), BERTScore (`roberta-large`, rescaled), deterministic numeric-match & citation scoring |
| Orchestration | plain Python (deterministic control flow), `ThreadPoolExecutor` for concurrent graph expansion |

## TigerGraph graph schema
- **Vertices:** `Company(ticker, name, sector)`, `Sector(name)`, `Filing(id, form, filing_date)`, `Chunk(id, text, ticker, form, doc_id, section, emb:VECTOR)`
- **Edges:** `Company —FILED→ Filing`, `Filing —HAS_CHUNK→ Chunk`, `Company —IN_SECTOR→ Sector`
- **Vector index:** native `ADD VECTOR ATTRIBUTE emb(DIMENSION=384, METRIC="COSINE")`; kNN via the built-in `vectorSearch` GSQL function.

## Ingestion & transformation
1. `round3_ingest.py` — parse 400 filings → section-aware, token-bounded chunks (~500 tok, 60 overlap) with metadata (ticker/form/date/section); build the graph backbone from `manifest.csv` ($0, offline).
2. `round3_load_tigergraph.py` — create schema + vector attribute, load Company/Sector/Filing/Chunk vertices + edges, upsert chunk embeddings (idempotent, resumable), wait for the HNSW index.

## Retrieval & context optimization (GraphRAG)
1. **Route** — extract query entities (companies / tickers / sectors) from the question.
2. **Vector seed** — high-recall `vectorSearch` over `Chunk.emb` in TigerGraph.
3. **2-hop graph fusion** — for routing targets **and** companies surfaced in the first-hop evidence (bridge entities), retrieve their filings concurrently (e.g. "company that acquired X → that company's auditor").
4. **Optimize (deterministic, 0 extra LLM tokens):** semantic **dedup** (keeps complementary facts), **MMR** diversity ranking, **graph-aware** boost, and **sentence-level compression** to an **adaptive budget** (tight for single-hop, larger for high-fan-out aggregation).
5. **Generate** — compact evidence → Gemini, with a `Sources: [ids]` citation line.

## Controls & fairness
- Same model, temperature (0), output cap, and one shared core system prompt across all pipelines.
- RAG and GraphRAG use the same embedding model.
- Judge is **evidence-blind** (grades answer vs reference only) — no pipeline advantaged by context length.
- Support & citations scored **deterministically** (reproducible).
- Held-out eval run **3×**; every result stamped with its **git commit**.
- Ablation isolates the contribution of graph / optimizer / compression / dedup / MMR.

## Repository map
```
backend/graph/   tigergraph_client.py · tigergraph_vector.py     (Savanna graph + vector)
backend/rag/     basic_rag_tg.py · graph_rag_tg.py · context_optimizer.py
backend/llm/     gemini_client.py                                 (shared prompts + LLM)
evaluation/      round3_benchmark.py                              (3 pipelines, token accounting, judge, ablation)
scripts/         round3_ingest.py · round3_load_tigergraph.py · round3_report.py · rejudge.py · query.py
data/            sp100_dataset/ · round3/round3_eval.json · round3/graph.json
results/round3/  run_*.json · summary.json · ablation.json · report.md/csv
```
