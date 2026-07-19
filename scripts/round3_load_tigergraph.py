#!/usr/bin/env python3
"""
Load the Round 3 SP100 graph + vectors into TigerGraph Savanna.

Creates the schema (Company, Sector, Filing, Chunk + native vector), loads the
graph backbone (FILED / HAS_CHUNK / IN_SECTOR) and upserts every chunk with its
embedding so BOTH vector search and graph traversal run inside TigerGraph.

Prereq:  python -m scripts.round3_ingest   (produces data/round3/{chunks.jsonl,graph.json})

Usage:   python -m scripts.round3_load_tigergraph
         python -m scripts.round3_load_tigergraph --schema-only
"""

import sys, json, argparse, logging
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.graph.tigergraph_client import TigerGraphClient
from backend.graph.tigergraph_vector import TigerGraphVectorStore, embed_batch, EMBED_DIM

load_dotenv(Path(__file__).parent.parent / ".env", override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
R3 = Path(__file__).parent.parent / "data" / "round3"


def setup_schema(conn, graph):
    # NOTE: exact vector-attribute + schema-change syntax is TigerGraph-4.2/Savanna
    # specific — confirm against your instance (see the Hybrid Graph+Vector doc).
    stmts = [
        f"""USE GRAPH {graph}
CREATE SCHEMA_CHANGE JOB r3_schema FOR GRAPH {graph} {{
  ADD VERTEX Company (PRIMARY_ID ticker STRING, name STRING, sector STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Sector (PRIMARY_ID name STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Filing (PRIMARY_ID id STRING, ticker STRING, form STRING, filing_date STRING) WITH primary_id_as_attribute="true";
  ADD VERTEX Chunk (PRIMARY_ID id STRING, text STRING, ticker STRING, form STRING, doc_id STRING, section STRING) WITH primary_id_as_attribute="true";
  ADD DIRECTED EDGE FILED (FROM Company, TO Filing);
  ADD DIRECTED EDGE HAS_CHUNK (FROM Filing, TO Chunk);
  ADD DIRECTED EDGE IN_SECTOR (FROM Company, TO Sector);
}}
RUN SCHEMA_CHANGE JOB r3_schema""",
        f"""USE GRAPH {graph}
CREATE SCHEMA_CHANGE JOB r3_vec FOR GRAPH {graph} {{
  ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE emb(DIMENSION={EMBED_DIM}, METRIC="COSINE");
}}
RUN SCHEMA_CHANGE JOB r3_vec""",
        f"""USE GRAPH {graph}
CREATE OR REPLACE QUERY chunk_search(LIST<float> query_vector, INT k) SYNTAX v3 {{
  MapAccum<Vertex,Float> @@dist;
  v = vectorSearch({{Chunk.emb}}, query_vector, k, {{distance_map:@@dist}});
  PRINT v; PRINT @@dist;
}}
INSTALL QUERY chunk_search""",
    ]
    for s in stmts:
        try:
            logger.info(conn.gsql(s))
        except Exception as e:
            logger.warning(f"schema stmt failed (continuing): {e}")


def load_graph(conn, g):
    g = json.load(open(R3 / "graph.json", encoding="utf-8"))
    conn.upsertVertices("Sector", [(s, {"name": s}) for s in g["sectors"]])
    conn.upsertVertices("Company", [(c["ticker"], {"name": c["name"], "sector": c["sector"]}) for c in g["companies"]])
    conn.upsertVertices("Filing", [(f["id"], {"ticker": f["ticker"], "form": f["form"], "filing_date": f["date"]}) for f in g["filings"]])
    by = lambda t: [e for e in g["edges"] if e["type"] == t]
    conn.upsertEdges("Company", "IN_SECTOR", "Sector", [(e["from"], e["to"], {}) for e in by("IN_SECTOR")])
    conn.upsertEdges("Company", "FILED", "Filing", [(e["from"], e["to"], {}) for e in by("FILED")])
    conn.upsertEdges("Filing", "HAS_CHUNK", "Chunk", [(e["from"], e["to"], {}) for e in by("HAS_CHUNK")])
    logger.info(f"[OK] graph backbone loaded: {len(g['companies'])} companies, {len(g['filings'])} filings")


def load_chunks(store, batch=200, skip=0):
    import time as _t
    rows = [json.loads(l) for l in open(R3 / "chunks.jsonl", encoding="utf-8")]
    logger.info(f"Embedding + upserting {len(rows)} chunks (starting at {skip}, batch={batch})...")
    for i in range(skip, len(rows), batch):
        part = rows[i:i + batch]
        vecs = embed_batch([c["text"] for c in part])
        verts = [(c["id"], {"text": c["text"][:60000], "ticker": c["ticker"], "form": c["form"],
                            "doc_id": c["id"].rsplit("#", 1)[0], "section": c.get("section", ""), "emb": v})
                 for c, v in zip(part, vecs)]
        for attempt in range(4):  # retry transient 499/timeout
            try:
                store.conn.upsertVertices("Chunk", verts); break
            except Exception as e:
                wait = 5 * (attempt + 1)
                logger.warning(f"  batch@{i} failed ({e}); retry {attempt+1}/4 in {wait}s")
                _t.sleep(wait)
        else:
            logger.error(f"  batch@{i} gave up — resume later with:  --only-chunks --skip {i}")
            return i
        logger.info(f"  {min(i+batch,len(rows))}/{len(rows)}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument("--only-chunks", action="store_true", help="skip schema+graph, just load chunks")
    ap.add_argument("--skip", type=int, default=0, help="resume chunk upsert from this index")
    args = ap.parse_args()
    tg = TigerGraphClient().connect()
    store = TigerGraphVectorStore(tg)
    if not args.only_chunks:
        setup_schema(tg.conn, tg.graph)
        if args.schema_only:
            logger.info("[DONE schema]"); return
        load_graph(tg.conn, tg.graph)
    done = load_chunks(store, skip=args.skip)
    if done >= 0:
        store.wait_for_index()
    logger.info("[DONE]")


if __name__ == "__main__":
    main()
