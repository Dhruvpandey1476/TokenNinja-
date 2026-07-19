"""
TigerGraph as a Vector Database — helper for Round 3.

Uses TigerGraph Savanna's native vector support (ADD VECTOR ATTRIBUTE + the
built-in vectorSearch function) to store chunk embeddings and run kNN search
*inside* TigerGraph — no FAISS, no external vector store.

This powers two things with the SAME embedding model (all-MiniLM-L6-v2, 384-dim),
which the Round 3 rules require:
  • Traditional RAG baseline  → pure vectorSearch (no graph traversal)
  • GraphRAG                  → vectorSearch to seed, then traverse the graph

Official syntax reference: TigerGraph "Hybrid Graph+Vector Search".
"""

import time
import logging
import requests as req_lib

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384                      # must equal the model's output dimension
VECTOR_METRIC = "COSINE"

_st_model = None


def get_embedder():
    """Lazily load the shared sentence-transformer (same model for RAG + GraphRAG)."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL)
        logger.info(f"[OK] Loaded embedder: {EMBED_MODEL} ({EMBED_DIM}-dim)")
    return _st_model


def embed(text: str) -> list[float]:
    return get_embedder().encode(text[:8191]).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in get_embedder().encode([t[:8191] for t in texts])]


class TigerGraphVectorStore:
    """Vector store backed by a TigerGraph vertex + native vector attribute."""

    def __init__(self, tg_client, vertex="Chunk", vec_attr="emb"):
        # tg_client: a connected backend.graph.tigergraph_client.TigerGraphClient
        self.tg = tg_client
        self.conn = tg_client.conn
        self.graph = tg_client.graph
        self.vertex = vertex
        self.vec_attr = vec_attr

    # ── One-time setup ────────────────────────────────────────────────────────
    def setup_schema(self, dimension: int = EMBED_DIM, metric: str = VECTOR_METRIC):
        """Create the Chunk vertex, add the vector attribute, install the search query.

        Idempotent-ish: 'already exists' errors are logged and ignored so this is
        safe to re-run.
        """
        g, v, a = self.graph, self.vertex, self.vec_attr

        def _gsql(stmt, label):
            try:
                logger.info(f"[SCHEMA] {label}")
                logger.info(self.conn.gsql(stmt))
            except Exception as e:
                logger.warning(f"[SCHEMA] {label} -> {e} (continuing)")

        # 1) Add the Chunk vertex to the graph
        _gsql(f"""USE GRAPH {g}
CREATE SCHEMA_CHANGE JOB add_{v.lower()} FOR GRAPH {g} {{
  ADD VERTEX {v} (PRIMARY_ID id STRING, text STRING, doc_id STRING, chunk_index INT)
      WITH primary_id_as_attribute="true";
}}
RUN SCHEMA_CHANGE JOB add_{v.lower()}""", f"add vertex {v}")

        # 2) Add the native vector attribute (HNSW index builds automatically)
        _gsql(f"""USE GRAPH {g}
CREATE SCHEMA_CHANGE JOB add_{a} FOR GRAPH {g} {{
  ALTER VERTEX {v} ADD VECTOR ATTRIBUTE {a}(DIMENSION={dimension}, METRIC="{metric}");
}}
RUN SCHEMA_CHANGE JOB add_{a}""", f"add vector attribute {a} (dim={dimension}, {metric})")

        # 3) Install the top-k vector-search query
        _gsql(f"""USE GRAPH {g}
CREATE OR REPLACE QUERY chunk_search(LIST<float> query_vector, INT k) SYNTAX v3 {{
  MapAccum<Vertex, Float> @@dist;
  v = vectorSearch({{{v}.{a}}}, query_vector, k, {{distance_map: @@dist}});
  PRINT v;
  PRINT @@dist;
}}
INSTALL QUERY chunk_search""", "install chunk_search query")

    # ── Ingest ────────────────────────────────────────────────────────────────
    def upsert_chunks(self, chunks: list[dict], batch: int = 500):
        """chunks: [{id, text, doc_id, chunk_index}] — embeds and stores the vector."""
        n = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vecs = embed_batch([c["text"] for c in part])
            rows = [(c["id"], {"text": c["text"][:60000], "doc_id": c.get("doc_id", ""),
                               "chunk_index": int(c.get("chunk_index", 0)),
                               self.vec_attr: vec})
                    for c, vec in zip(part, vecs)]
            try:
                self.conn.upsertVertices(self.vertex, rows)
                n += len(rows)
                logger.info(f"[VEC] upserted {n}/{len(chunks)} chunks")
            except Exception as e:
                logger.error(f"[VEC] upsert batch failed: {e}")
        return n

    # ── Index readiness (HNSW indexing is asynchronous) ───────────────────────
    def wait_for_index(self, timeout: int = 300, poll: int = 5) -> bool:
        """Poll /restpp/vector/status until Ready_for_query (index lags ingest)."""
        url = f"https://{self.tg.host}:{self.tg.port}/restpp/vector/status"
        hdr = getattr(self.conn, "authHeader", {}) or {}
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = req_lib.get(url, headers=hdr, timeout=15)
                if "Ready_for_query" in r.text:
                    logger.info("[VEC] index Ready_for_query")
                    return True
                logger.info(f"[VEC] index status: {r.text[:80]}")
            except Exception as e:
                logger.warning(f"[VEC] status check failed: {e}")
            time.sleep(poll)
        logger.warning("[VEC] index not confirmed ready within timeout")
        return False

    # ── Search (pure vector, no graph) ────────────────────────────────────────
    def vector_search(self, question: str, k: int = 5) -> list[dict]:
        """Return top-k chunks: [{id, text, doc_id, score}] — for RAG + citations."""
        qvec = embed(question)
        res = self.conn.runInstalledQuery("chunk_search", {"query_vector": qvec, "k": k})
        verts, dists = [], {}
        for block in res or []:
            if "v" in block:
                verts = block["v"]
            if "@@dist" in block:
                dists = block["@@dist"]
        out = []
        for vt in verts:
            attrs = vt.get("attributes", vt)
            vid = vt.get("v_id", attrs.get("id"))
            out.append({
                "id": vid,
                "text": attrs.get("text", ""),
                "doc_id": attrs.get("doc_id", ""),
                "score": dists.get(vid),
            })
        return out
