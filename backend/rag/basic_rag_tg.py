"""
Traditional RAG baseline — retrieval powered by TigerGraph's native vector DB
(no graph traversal). Round 3 version.

Same embedding model as GraphRAG (all-MiniLM-L6-v2), TigerGraph Savanna as the
vector store. Returns citations (chunk + doc ids) as required by Round 3.
"""

import time
import logging
from dataclasses import dataclass, field

from ..graph.tigergraph_vector import TigerGraphVectorStore
from ..llm.gemini_client import gemini_generate, MAX_OUTPUT_TOKENS, ROUND3_SYSTEM_CORE, ROUND3_EVIDENCE_CLAUSE

logger = logging.getLogger(__name__)

TOP_K = 8  # good-faith baseline: enough recall to be competitive (sensible, disclosed)


@dataclass
class RAGTGResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    context_tokens: int = 0
    retrieved_ids: list = field(default_factory=list)   # chunk ids (citations)
    retrieved_docs: list = field(default_factory=list)  # doc ids (citations)
    evidence: str = ""                                  # actual context sent to the LLM
    method: str = "rag_tigergraph_vector"


class BasicRAGTigerGraph:
    """Vector-only RAG using TigerGraph Savanna as the vector database."""

    def __init__(self, tg_client, top_k: int = TOP_K):
        self.store = TigerGraphVectorStore(tg_client)
        self.top_k = top_k

    def query(self, question: str, top_k: int = None) -> RAGTGResult:
        t0 = time.time()
        k = top_k or self.top_k

        # 1. Retrieve top-k chunks via TigerGraph vector search (no graph traversal)
        hits = self.store.vector_search(question, k=k)
        context = "\n\n---\n\n".join(h["text"] for h in hits if h.get("text"))

        # 2. Same prompt policy as the other pipelines (context as support, no refusal)
        system_prompt = ROUND3_SYSTEM_CORE + ROUND3_EVIDENCE_CLAUSE
        user_prompt = (
            f"Question: {question}\n\n"
            f"Evidence (top {k} passages):\n{context}\n\n"
            "Answer (cite ids):"
        )

        result = gemini_generate(system_prompt=system_prompt, user_prompt=user_prompt,
                                 temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)

        from ..llm.gemini_client import count_context_tokens
        return RAGTGResult(
            answer=result["answer"].strip(),
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
            latency_ms=(time.time() - t0) * 1000,
            context_tokens=count_context_tokens(context),
            retrieved_ids=[h["id"] for h in hits],
            retrieved_docs=sorted({h["doc_id"] for h in hits if h.get("doc_id")}),
            evidence=context,
        )
