"""
TigerGraph Client — wraps pyTigerGraph for GraphRAG operations.
Handles connection, schema setup, ingestion, and subgraph queries.
"""

import os
import re
import json
import logging
from typing import Optional
import requests as req_lib
import pyTigerGraph as tg
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Cap how many nodes we expand per hop. One getEdges REST call is made per
# frontier node, so bounding the frontier bounds latency (robust to a slow/
# variable TigerGraph cloud instance) while keeping genuine multi-hop traversal.
# Serialization only uses the top few entities/relationships anyway.
MAX_FRONTIER = int((os.getenv("MAX_FRONTIER", "6") or "6").strip())


class TigerGraphClient:
    """Production-ready TigerGraph client for GraphRAG operations."""

    def __init__(self):
        self.host = (os.getenv("TIGERGRAPH_HOST", "localhost") or "").strip()
        self.graph = (os.getenv("TIGERGRAPH_GRAPH", "GraphRAGDemo") or "").strip()
        self.username = (os.getenv("TIGERGRAPH_USERNAME", "tigergraph") or "").strip()
        self.password = (os.getenv("TIGERGRAPH_PASSWORD", "tigergraph") or "").strip()
        self.secret = (os.getenv("TIGERGRAPH_SECRET", "") or "").strip()
        self.port = int((os.getenv("TIGERGRAPH_PORT", "443") or "443").strip())
        self.use_ssl = (os.getenv("TIGERGRAPH_USE_SSL", "true") or "true").lower().strip() == "true"
        self.conn: Optional[tg.TigerGraphConnection] = None

    def connect(self) -> "TigerGraphClient":
        """Establish connection and get auth token."""
        try:
            url = f"https://{self.host}" if self.use_ssl else f"http://{self.host}"
            logger.info(f"Connecting to TigerGraph at {url}...")
            logger.info(f"Graph: {self.graph}, Username: {self.username}")
            
            self.conn = tg.TigerGraphConnection(
                host=url,
                graphname=self.graph,
                username=self.username,
                password=self.password,
                restppPort=self.port,
                gsPort=self.port,
                tgCloud=True,
            )
            logger.info(f"[OK] TigerGraph connection object created")
            
            # Try to get token via secret first (TG 3.x)
            if self.secret:
                logger.info(f"Attempting to authenticate with secret...")
                try:
                    token = self.conn.getToken(self.secret)
                    logger.info(f"[OK] Authentication successful - token obtained")
                    return self
                except Exception as token_err:
                    logger.warning(f"[WARN]  Secret-based auth failed: {token_err}")
            
            # Fallback: TigerGraph 4.x Savanna — POST /gsql/v1/tokens with username/password
            logger.info("Attempting TigerGraph 4.x token auth via /gsql/v1/tokens...")
            try:
                token_url = f"{url}:{self.port}/gsql/v1/tokens"
                resp = req_lib.post(
                    token_url,
                    json={"graph": self.graph, "lifetime": "2592000000"},
                    auth=(self.username, self.password),
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    jwt_token = data.get("token")
                    if jwt_token:
                        self.conn.apiToken = jwt_token
                        self.conn.authHeader = {'Authorization': 'Bearer ' + jwt_token}
                        logger.info(f"[OK] JWT token obtained via TG 4.x auth")
                else:
                    logger.warning(f"[WARN]  TG 4.x token endpoint returned {resp.status_code}: {resp.text[:100]}")
            except Exception as v4_err:
                logger.warning(f"[WARN]  TG 4.x token auth failed: {v4_err}")
                logger.warning("Continuing without token (may limit some operations)")
            
            logger.info(f"[OK] Connected to TigerGraph: {self.host}/{self.graph}")
            return self
        except Exception as e:
            logger.error(f"[ERR] TigerGraph connection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    # ─── Schema Setup ──────────────────────────────────────────────

    def create_schema(self):
        """Create the knowledge graph schema via GSQL."""
        schema_gsql = """
        USE GLOBAL
        
        CREATE VERTEX Entity (
            PRIMARY_ID entity_id STRING,
            name STRING,
            entity_type STRING,
            description STRING,
            embedding LIST<DOUBLE>,
            doc_source STRING,
            created_at DATETIME
        ) WITH primary_id_as_attribute="true"

        CREATE VERTEX Document (
            PRIMARY_ID doc_id STRING,
            title STRING,
            content STRING,
            chunk_index INT,
            token_count INT,
            source_url STRING,
            created_at DATETIME
        ) WITH primary_id_as_attribute="true"

        CREATE VERTEX Concept (
            PRIMARY_ID concept_id STRING,
            name STRING,
            category STRING,
            importance_score FLOAT
        ) WITH primary_id_as_attribute="true"

        CREATE UNDIRECTED EDGE RELATED_TO (
            FROM Entity, TO Entity,
            relation_type STRING,
            confidence FLOAT,
            context STRING
        )

        CREATE DIRECTED EDGE MENTIONED_IN (
            FROM Entity, TO Document,
            frequency INT,
            relevance_score FLOAT
        )

        CREATE DIRECTED EDGE HAS_CONCEPT (
            FROM Document, TO Concept,
            weight FLOAT
        )

        CREATE DIRECTED EDGE CO_OCCURS_WITH (
            FROM Entity, TO Entity,
            co_occurrence_count INT,
            documents LIST<STRING>
        )

        CREATE GRAPH GraphRAGDemo (
            Entity, Document, Concept,
            RELATED_TO, MENTIONED_IN, HAS_CONCEPT, CO_OCCURS_WITH
        )
        """
        result = self.conn.gsql(schema_gsql)
        logger.info(f"Schema created: {result}")
        return result

    # ─── Ingestion ─────────────────────────────────────────────────

    def upsert_entity(self, entity_id: str, name: str, entity_type: str,
                      description: str, embedding: list, doc_source: str):
        """Insert or update an entity vertex."""
        attributes = {
            "name": name,
            "entity_type": entity_type,
            "description": description,
        }
        
        # Try to add optional fields if schema supports them
        if embedding:
            try:
                attributes["embedding"] = embedding
            except:
                pass  # Schema may not support embeddings
        
        if doc_source:
            try:
                attributes["doc_source"] = doc_source
            except:
                pass  # Schema may not support doc_source
        
        try:
            self.conn.upsertVertex(
                "Entity", entity_id,
                attributes=attributes
            )
        except Exception as e:
            # Try with fewer attributes if it fails
            if "Unknown vertex attribute" in str(e):
                logger.debug(f"Some attributes not in schema, retrying with minimal attributes")
                minimal_attrs = {
                    "name": name,
                    "entity_type": entity_type,
                }
                try:
                    self.conn.upsertVertex(
                        "Entity", entity_id,
                        attributes=minimal_attrs
                    )
                except Exception as e2:
                    logger.warning(f"Could not upsert entity even with minimal attributes: {e2}")
                    raise
            else:
                raise

    def upsert_document(self, doc_id: str, title: str, content: str,
                        chunk_index: int, token_count: int, source_url: str = ""):
        """Insert or update a document vertex."""
        self.conn.upsertVertex(
            "Document", doc_id,
            attributes={
                "title": title,
                "content": content,
                "chunk_index": chunk_index,
                "token_count": token_count,
                "source_url": source_url,
            }
        )

    def upsert_relationship(self, from_entity: str, to_entity: str,
                             relation_type: str, confidence: float, context: str):
        """Insert or update a RELATED_TO edge."""
        self.conn.upsertEdge(
            "Entity", from_entity,
            "RELATED_TO",
            "Entity", to_entity,
            attributes={
                "relation_type": relation_type,
                "confidence": confidence,
                "context": context,
            }
        )

    def link_entity_to_document(self, entity_id: str, doc_id: str,
                                 frequency: int, relevance_score: float):
        """Link entity to document via MENTIONED_IN edge."""
        self.conn.upsertEdge(
            "Entity", entity_id,
            "MENTIONED_IN",
            "Document", doc_id,
            attributes={
                "frequency": frequency,
                "relevance_score": relevance_score,
            }
        )

    # ─── Graph Retrieval ───────────────────────────────────────────

    def is_empty(self) -> bool:
        """Fast check: is the graph empty? Used to skip expensive entity extraction."""
        try:
            if not self.conn:
                logger.warning(f"is_empty(): conn is None, graph is EMPTY")
                return True
            # Quick check: get vertex count
            result = self.conn.getVertexCount("Entity")
            is_empty = result == 0
            logger.info(f"is_empty() check: Entity count = {result}, is_empty = {is_empty}")
            return is_empty
        except Exception as e:
            logger.warning(f"is_empty() check failed: {e}, assuming graph is EMPTY")
            return True

    def get_entity_subgraph(self, entity_names: list[str], max_hops: int = 2,
                             max_neighbors: int = 10,
                             include_documents: bool = False) -> dict:
        """
        Core GraphRAG retrieval: given seed entities from the query,
        traverse the graph up to ``max_hops`` and return the subgraph context.

        Returns structured context (entities + relationships) instead of raw text
        chunks — this is what enables the large token reduction.

        ``include_documents`` is OFF by default: the answer prompt is built from
        entities + relationship triples only, so fetching MENTIONED_IN documents
        would add round-trips (latency) without affecting the prompt. Enable it
        only when you need source documents for inspection.
        """
        if not entity_names:
            return {"entities": [], "relationships": [], "documents": []}

        logger.info(f"Traversing graph from seed entities: {entity_names} (max_hops={max_hops}, max_neighbors={max_neighbors})")

        try:
            if not self.conn:
                logger.error("TigerGraph connection not available")
                raise Exception("No TigerGraph connection")

            return self._traverse_subgraph(entity_names, max_hops, max_neighbors,
                                           include_documents)
        except Exception as e:
            logger.error(f"[ERR] TigerGraph retrieval failed: {e}")
            raise

    @staticmethod
    def _normalize_entity_id(name: str) -> str:
        """Map an entity name to its vertex primary_id.

        Must match DocumentIngestionPipeline._make_entity_id exactly so query
        seeds resolve to the same vertices that were written at ingest time.
        """
        return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())

    def _fetch_entity(self, entity_id: str) -> Optional[dict]:
        """Fetch a single Entity vertex by its primary_id (no full-graph scan)."""
        try:
            res = self.conn.getVerticesById("Entity", entity_id)
            if res:
                return res[0] if isinstance(res, list) else res
        except Exception as e:
            logger.debug(f"getVerticesById failed for '{entity_id}': {e}")
        return None

    def _find_entity_by_name(self, name: str) -> Optional[dict]:
        """Fallback seed resolution: filter Entity vertices by the name attribute.

        Used only when the normalized-id lookup misses (e.g. the ingest-time id
        scheme differed). Still a targeted server-side filter — NOT a full scan.
        """
        try:
            safe = name.replace('"', '').strip()
            res = self.conn.getVertices("Entity", where=f'name="{safe}"', limit=1)
            if res:
                return res[0] if isinstance(res, list) else res
        except Exception as e:
            logger.debug(f"name-attribute lookup failed for '{name}': {e}")
        return None

    def _resolve_seeds(self, entity_names: list[str]) -> dict:
        """Resolve query entity names to seed Entity vertices, keyed by v_id."""
        seeds = {}
        for name in entity_names:
            eid = self._normalize_entity_id(name)
            vertex = self._fetch_entity(eid)
            if vertex is None:
                vertex = self._find_entity_by_name(name)
            if vertex and vertex.get("v_id") and vertex["v_id"] not in seeds:
                seeds[vertex["v_id"]] = vertex
                logger.info(f"  [CHECK] Seed '{name}' → vertex '{vertex['v_id']}'")
            else:
                logger.debug(f"  [BAD] Seed '{name}' not found in graph")
        return seeds

    def _traverse_subgraph(self, entity_names: list[str], max_hops: int,
                           max_neighbors: int, include_documents: bool = False) -> dict:
        """Real multi-hop BFS over the knowledge graph.

        Starting from the resolved seed vertices, expand the frontier hop by hop
        (up to ``max_hops``) along RELATED_TO edges, collecting neighbor entities
        and the relationships that connect them. Each vertex contributes at most
        ``max_neighbors`` edges per hop. Documents are pulled for the seed
        entities via MENTIONED_IN. This is genuine graph traversal — no
        whole-graph fetch, no Python fuzzy matching.
        """
        entities_by_id: dict = {}
        relationships: list = []
        rel_seen: set = set()
        documents: list = []
        doc_seen: set = set()

        # Hop 0: seed vertices
        seeds = self._resolve_seeds(entity_names)
        if not seeds:
            logger.warning("No seed entities resolved in graph")
            return {"entities": [], "relationships": [], "documents": []}

        entities_by_id.update(seeds)
        visited: set = set(seeds.keys())
        frontier: list = list(seeds.keys())

        # Hops 1..max_hops: expand the frontier along RELATED_TO edges
        for hop in range(max_hops):
            if not frontier:
                break
            next_frontier: list = []
            # Cap frontier expansion to bound the number of getEdges calls (and
            # thus latency) — still a real second hop, just not an unbounded fan-out.
            for vid in frontier[:MAX_FRONTIER]:
                try:
                    edges = self.conn.getEdges("Entity", vid, "RELATED_TO") or []
                except Exception as e:
                    logger.debug(f"getEdges failed for '{vid}': {e}")
                    edges = []

                for edge in edges[:max_neighbors]:
                    from_id = edge.get("from_id")
                    to_id = edge.get("to_id")
                    neighbor_id = to_id if from_id == vid else from_id

                    # Record the relationship (dedup undirected edges)
                    rel_key = tuple(sorted([str(from_id), str(to_id)])) + (edge.get("e_type", "RELATED_TO"),)
                    if rel_key not in rel_seen:
                        rel_seen.add(rel_key)
                        relationships.append(edge)

                    # Enqueue unvisited neighbors for the next hop.
                    # Build the neighbor vertex LOCALLY from its id (which is the
                    # normalized entity name) instead of a per-neighbor
                    # getVerticesById REST call — that fetch was the dominant
                    # latency cost (30-50 sequential round-trips/query). The id
                    # already yields a readable name (e.g. "knowledge_graph" ->
                    # "knowledge graph"), which is all serialization needs.
                    if neighbor_id and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        entities_by_id[neighbor_id] = {
                            "v_id": neighbor_id,
                            "attributes": {"name": neighbor_id.replace("_", " "),
                                           "entity_type": ""},
                        }
                        next_frontier.append(neighbor_id)
            logger.info(f"  Hop {hop + 1}: frontier {len(frontier)} → {len(next_frontier)} new entities")
            frontier = next_frontier

        # Optionally pull supporting documents for the seed entities (1-hop
        # MENTIONED_IN). OFF by default — the prompt uses triples only, so this
        # is pure extra latency unless documents are explicitly requested.
        if include_documents:
            for vid in seeds:
                try:
                    doc_edges = self.conn.getEdges("Entity", vid, "MENTIONED_IN") or []
                except Exception as e:
                    logger.debug(f"getEdges(MENTIONED_IN) failed for '{vid}': {e}")
                    doc_edges = []
                for edge in doc_edges[:max_neighbors]:
                    doc_id = edge.get("to_id")
                    if doc_id and doc_id not in doc_seen:
                        doc_seen.add(doc_id)
                        try:
                            dv = self.conn.getVerticesById("Document", doc_id)
                            if dv:
                                documents.append(dv[0] if isinstance(dv, list) else dv)
                        except Exception as e:
                            logger.debug(f"Could not fetch document '{doc_id}': {e}")

        logger.info(
            f"[OK] Traversal complete: {len(entities_by_id)} entities, "
            f"{len(relationships)} relationships, {len(documents)} documents "
            f"across {max_hops} hop(s)"
        )
        return {
            "entities": list(entities_by_id.values()),
            "relationships": relationships,
            "documents": documents,
        }

    def _parse_subgraph_result(self, raw_result) -> dict:
        """Parse GSQL result into structured subgraph dict."""
        if isinstance(raw_result, str):
            try:
                raw_result = json.loads(raw_result)
            except Exception:
                return {"entities": [], "relationships": [], "documents": []}

        entities = []
        relationships = []
        documents = []

        if isinstance(raw_result, list):
            for block in raw_result:
                if "entities" in block:
                    entities = block["entities"]
                if "relationships" in block:
                    relationships = block["relationships"]
                if "documents" in block:
                    documents = block["documents"]

        return {
            "entities": entities,
            "relationships": relationships,
            "documents": documents,
        }

    def get_stats(self) -> dict:
        """Return graph statistics."""
        try:
            vertex_counts = self.conn.getVertexCount()
            edge_counts = self.conn.getEdgeCount()
            return {
                "vertices": vertex_counts,
                "edges": edge_counts,
                "graph": self.graph,
                "host": self.host,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
