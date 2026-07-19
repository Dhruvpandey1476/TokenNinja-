"""
Optimized TigerGraph GraphRAG (Round 3).

Flow:  question -> route to graph entities (company/sector) -> high-recall vector
seed from TigerGraph -> graph expansion (pull chunks for graph-selected companies)
-> context optimizer (dedup+MMR+graph boost) -> compact evidence -> LLM.

Vector search + graph relationships both live in TigerGraph Savanna.
Returns citations + token breakdown + ablation-friendly toggles.
"""

import re
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

from ..graph.tigergraph_vector import TigerGraphVectorStore
from ..rag.context_optimizer import optimize, compress
from ..llm.gemini_client import gemini_generate, MAX_OUTPUT_TOKENS, ROUND3_SYSTEM_CORE, ROUND3_EVIDENCE_CLAUSE, count_context_tokens

logger = logging.getLogger(__name__)
META = Path(__file__).parent.parent.parent / "data" / "round3" / "graph.json"


@dataclass
class GraphRAGTGResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    context_tokens: int = 0
    extra_llm_tokens: int = 0          # tokens for any LLM optimization step (0 = deterministic)
    retrieved_ids: list = field(default_factory=list)
    retrieved_docs: list = field(default_factory=list)
    evidence: str = ""
    used_graph: bool = True
    target_entities: list = field(default_factory=list)
    method: str = "graphrag_tigergraph"


class GraphRAGTG:
    def __init__(self, tg_client, meta_path=META):
        self.store = TigerGraphVectorStore(tg_client)
        meta = json.load(open(meta_path, encoding="utf-8")) if Path(meta_path).exists() else {"companies": [], "sectors": []}
        self.companies = meta.get("companies", [])
        self.sectors = meta.get("sectors", [])
        # lookup maps for routing
        self.name2ticker = {c["name"].lower(): c["ticker"] for c in self.companies}
        self.ticker_set = {c["ticker"] for c in self.companies}
        self.sector2tickers = {}
        for c in self.companies:
            self.sector2tickers.setdefault(c["sector"].lower(), []).append(c["ticker"])

    def _route(self, q: str):
        """Graph routing: which companies/sectors does the question target?"""
        ql = q.lower()
        targets = set()
        for name, tk in self.name2ticker.items():
            if name and name.split()[0] in ql:  # first token of company name
                targets.add(tk)
        for tk in self.ticker_set:
            if re.search(rf"\b{re.escape(tk)}\b", q):
                targets.add(tk)
        for sec, tks in self.sector2tickers.items():
            if sec and sec in ql:
                targets.update(tks)  # sector aggregation → all its companies
        return targets

    def query(self, question, k_seed=40, top_n=8, use_graph=True, use_optimizer=True,
              use_dedup=True, use_mmr=True, use_compress=True, context_budget=1000):
        t0 = time.time()
        targets = self._route(question) if use_graph else set()

        # high-recall seed from TigerGraph vector search
        hits = self.store.vector_search(question, k=k_seed)
        for h in hits:
            h["graph_hit"] = use_graph and (h.get("id", "").split(":")[0] in targets)

        # 2-hop graph fusion + aggregation coverage. The companies to expand on are:
        #   (a) routing targets (named companies / sectors in the question), and
        #   (b) BRIDGE companies surfaced in the first-hop evidence (e.g. "AbbVie
        #       acquired Apogee" -> then fetch AbbVie's other filings for step 2).
        # Each gets a focused second retrieval, run concurrently (~one round-trip).
        if use_graph:
            bridge = {h["id"].split(":")[0] for h in hits[:6] if h.get("id")}
            expand = list((set(targets) | bridge))[:12]
            if expand:
                from concurrent.futures import ThreadPoolExecutor

                def _fetch(tk):
                    r = self.store.vector_search(f"{tk} {question}", k=3)
                    for h in r:
                        h["graph_hit"] = True
                    return r

                with ThreadPoolExecutor(max_workers=min(12, len(expand))) as ex:
                    for r in ex.map(_fetch, expand):
                        hits.extend(r)

        # drop exact-duplicate chunks (first + second hop overlap), keep first seen
        seen, uniq = set(), []
        for h in hits:
            if h.get("id") in seen:
                continue
            seen.add(h.get("id")); uniq.append(h)
        hits = uniq

        # optimize (deterministic) or plain top-n
        if use_optimizer:
            sel, _ = optimize(hits, question, top_n=top_n, use_dedup=use_dedup,
                              use_mmr=use_mmr, use_graph=use_graph)
        else:
            sel = hits[:top_n]

        # Adaptive budget (endorsed by the challenge): single-hop stays tight,
        # high-fan-out aggregation gets room to hold one fact per target company.
        eff_budget = context_budget
        if use_graph and targets:
            eff_budget = min(2200, context_budget + 120 * len(targets))

        # Deterministic sentence-level compression to the budget — 0 extra LLM tokens.
        if use_compress:
            context, cited = compress(sel, question, max_tokens=eff_budget)
        else:
            context = "\n\n---\n\n".join(f"[{h.get('id')}] {h['text']}" for h in sel if h.get("text"))
            cited = [h["id"].rsplit("#", 1)[0] for h in sel if h.get("id")]
        system = ROUND3_SYSTEM_CORE + ROUND3_EVIDENCE_CLAUSE
        user = f"Question: {question}\n\nEvidence:\n{context}\n\nAnswer (cite ids):"
        r = gemini_generate(system_prompt=system, user_prompt=user, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)

        return GraphRAGTGResult(
            answer=r["answer"].strip(),
            prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
            total_tokens=r["total_tokens"], latency_ms=(time.time() - t0) * 1000,
            context_tokens=count_context_tokens(context), extra_llm_tokens=0,
            retrieved_ids=[h.get("id") for h in sel],
            retrieved_docs=sorted({c.rsplit("#", 1)[0] for c in cited}),
            evidence=context,
            used_graph=use_graph, target_entities=sorted(targets),
        )
