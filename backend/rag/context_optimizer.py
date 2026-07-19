"""
Context optimizer — turns high-recall GraphRAG retrieval into a small, precise,
non-redundant evidence set. Fully DETERMINISTIC (no LLM) → adds ZERO inference
tokens, which is a deliberate advantage for total-inference-token accounting.

Three mechanisms (each independently toggleable for ablation):
  1. dedup   — drop near-duplicate passages BUT keep complementary ones
               (a near-dupe that introduces new numbers/dates/entities is kept).
  2. mmr     — Maximal Marginal Relevance: balance query-relevance vs diversity.
  3. graph   — boost passages from graph-selected companies/sectors.
"""

import re
import numpy as np
from ..graph.tigergraph_vector import embed, embed_batch

_NUM = re.compile(r"\$?\d[\d,\.]*\s?(?:million|billion|bn|m|b|%|k)?", re.I)


def _norm(m):
    m = np.asarray(m, dtype="float32")
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(n, 1e-8, None)


def _facts(text):
    return set(re.findall(_NUM, text)[:12])


def optimize(candidates, query, top_n=6, dedup_thresh=0.90, mmr_lambda=0.6,
             graph_boost=0.15, use_dedup=True, use_mmr=True, use_graph=True):
    """candidates: [{id,text,doc_id,score,graph_hit(bool)}] -> (selected, dropped)."""
    if not candidates:
        return [], []
    texts = [c["text"] for c in candidates]
    E = _norm(embed_batch(texts))
    q = _norm(embed(query))
    rel = E @ q  # cosine relevance to the query

    if use_graph:  # graph-derived boost: passages from graph-selected entities
        rel = rel + np.array([graph_boost if c.get("graph_hit") else 0.0 for c in candidates], dtype="float32")

    order = list(np.argsort(-rel))

    # 1) dedup — greedy, keep complementary (new numeric facts survive)
    if use_dedup:
        kept, kept_facts, dropped = [], set(), []
        for i in order:
            dup = any(float(E[i] @ E[j]) >= dedup_thresh for j in kept)
            if dup and not (_facts(texts[i]) - kept_facts):
                dropped.append(i); continue
            kept.append(i); kept_facts |= _facts(texts[i])
        order = kept
    else:
        dropped = []

    # 2) MMR selection of top_n (diversity-aware)
    if use_mmr:
        selected, pool = [], order[:]
        while pool and len(selected) < top_n:
            if not selected:
                best = pool[0]
            else:
                best, best_s = None, -1e9
                for i in pool:
                    div = max(float(E[i] @ E[j]) for j in selected)
                    s = mmr_lambda * float(rel[i]) - (1 - mmr_lambda) * div
                    if s > best_s:
                        best_s, best = s, i
            selected.append(best); pool.remove(best)
        chosen = selected
    else:
        chosen = order[:top_n]

    sel = [candidates[i] for i in chosen]
    drp = [candidates[i] for i in order[top_n:]] + [candidates[i] for i in dropped]
    return sel, drp


_SENT = re.compile(r"(?<=[.!?])\s+")


def compress(chunks, query, max_tokens=700):
    """Deterministic sentence-level compression (0 LLM tokens).

    Splits selected chunks into sentences, keeps the ones most relevant to the
    query (embedding cosine) up to a token budget. This is what makes GraphRAG
    cheaper than RAG: the graph finds the right chunks, this keeps only the
    answer-bearing sentences. Returns a compact context string + kept chunk ids.
    """
    sents = []
    for c in chunks:
        for s in _SENT.split(c.get("text", "")):
            s = s.strip()
            if len(s) > 25:
                sents.append((c.get("id"), s))
    if not sents:
        return "", []
    E = _norm(embed_batch([s for _, s in sents]))
    q = _norm(embed(query))
    qterms = {w for w in re.findall(r"[a-z0-9]{4,}", query.lower())}
    scores = []
    for k, (cid, s) in enumerate(sents):
        emb = float(E[k] @ q)
        sl = s.lower()
        lex = (len([1 for w in qterms if w in sl]) / len(qterms)) if qterms else 0.0
        has_num = bool(_NUM.search(s))
        num = 0.15 if has_num else 0.0                   # keep sentences with figures
        combo = 0.25 if (has_num and lex > 0) else 0.0   # "answer sentence": figure + query term
        scores.append(emb + 0.40 * lex + num + combo)
    order = list(np.argsort(-np.array(scores)))

    def _tok(s): return max(1, len(s) // 4)
    kept, ids, tok = [], set(), 0
    # pass 1: guarantee the best sentence from each distinct chunk (multi-doc coverage)
    seen_best = set()
    for i in order:
        cid, s = sents[int(i)]
        if cid in seen_best:
            continue
        seen_best.add(cid)
        if tok + _tok(s) > max_tokens:
            continue
        kept.append((int(i), cid, s)); ids.add(cid); tok += _tok(s)
    # pass 2: fill remaining budget with next-best sentences overall
    kept_idx = {k for k, _, _ in kept}
    for i in order:
        if int(i) in kept_idx:
            continue
        cid, s = sents[int(i)]
        if tok + _tok(s) > max_tokens:
            continue
        kept.append((int(i), cid, s)); ids.add(cid); tok += _tok(s)
    kept.sort(key=lambda x: x[0])  # restore reading order
    ctx = "\n".join(f"[{cid}] {s}" for _, cid, s in kept)
    return ctx, sorted(ids)
