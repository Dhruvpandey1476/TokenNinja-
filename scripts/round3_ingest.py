#!/usr/bin/env python3
"""
Round 3 ingestion — SP100 SEC filings -> metadata-rich chunks + graph model.

Offline, $0 (no LLM). Produces two artifacts that feed the pipelines:
  data/round3/chunks.jsonl  — one line per chunk: {id, ticker, form, date, section, text, n_tokens}
  data/round3/graph.json    — {companies, sectors, filings, edges} graph backbone

Graph backbone (meaningful graph use, cheaply, from manifest.csv):
  Company(ticker, name, sector) -[FILED]-> Filing(id, form, date)
  Filing -[HAS_CHUNK]-> Chunk
  Company -[IN_SECTOR]-> Sector
This backbone alone powers the multi-hop / sector-aggregation questions
(e.g. "which Information Technology companies ...") without any extraction cost.

Usage:
    python -m scripts.round3_ingest              # full corpus
    python -m scripts.round3_ingest --limit 5    # quick test on 5 companies
"""

import re
import csv
import json
import argparse
import logging
from pathlib import Path

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "sp100_dataset"
OUT = ROOT / "data" / "round3"
ENC = tiktoken.get_encoding("cl100k_base")

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 60
# SEC section anchors — split long filings on these so chunks are section-coherent.
SECTION_RE = re.compile(r"(?im)^\s*(item\s+\d+[a-z]?\.?.*|part\s+[ivx]+.*)$")


def n_tok(text: str) -> int:
    return len(ENC.encode(text))


def chunk_text(text: str):
    """Section-aware, token-bounded chunks with overlap. Yields (section, text)."""
    # Split into (section_label, body) segments on SEC item/part headers.
    segments, cur_label, buf = [], "", []
    for line in text.splitlines():
        if SECTION_RE.match(line.strip()) and len(line.strip()) < 120:
            if buf:
                segments.append((cur_label, "\n".join(buf)))
            cur_label, buf = line.strip()[:80], []
        else:
            buf.append(line)
    if buf:
        segments.append((cur_label, "\n".join(buf)))

    for label, body in segments:
        toks = ENC.encode(body)
        if not toks:
            continue
        step = CHUNK_TOKENS - CHUNK_OVERLAP
        for i in range(0, len(toks), step):
            piece = ENC.decode(toks[i:i + CHUNK_TOKENS]).strip()
            if len(piece) > 40:
                yield label, piece


def load_manifest():
    """ticker -> {name, sector}; and (ticker, filename)-> {form, date}."""
    companies, filings = {}, {}
    mpath = DATA / "manifest.csv"
    if not mpath.exists():
        return companies, filings
    for row in csv.DictReader(open(mpath, encoding="utf-8")):
        t = row["ticker"].strip()
        companies[t] = {"ticker": t, "name": row["name"].strip(), "sector": row["sector"].strip()}
        fname = Path(row["local_path"]).name
        filings[(t, fname)] = {"form": row["form"].strip().replace(" ", ""), "date": row["filing_date"].strip()}
    return companies, filings


def parse_form_date(fname: str):
    m = re.match(r"([A-Z0-9-]+)_(\d{4}-\d{2}-\d{2})", fname)
    return (m.group(1), m.group(2)) if m else ("UNKNOWN", "")


def main(limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    companies_meta, filings_meta = load_manifest()

    tickers = sorted([d.name for d in DATA.iterdir() if d.is_dir()])
    if limit:
        tickers = tickers[:limit]

    companies, sectors, filings, edges = {}, set(), {}, []
    chunk_f = open(OUT / "chunks.jsonl", "w", encoding="utf-8")
    n_chunks = 0

    for tk in tickers:
        meta = companies_meta.get(tk, {"ticker": tk, "name": tk, "sector": "Unknown"})
        companies[tk] = meta
        sectors.add(meta["sector"])
        edges.append({"from": tk, "to": meta["sector"], "type": "IN_SECTOR"})

        for fp in sorted((DATA / tk).glob("*.txt")):
            form, date = filings_meta.get((tk, fp.name), {"form": "", "date": ""}).values() \
                if (tk, fp.name) in filings_meta else parse_form_date(fp.name)
            fid = f"{tk}:{fp.name[:-4]}"
            filings[fid] = {"id": fid, "ticker": tk, "form": form, "date": date}
            edges.append({"from": tk, "to": fid, "type": "FILED"})

            text = fp.read_text(encoding="utf-8", errors="ignore")
            for ci, (section, piece) in enumerate(chunk_text(text)):
                cid = f"{fid}#c{ci}"
                chunk_f.write(json.dumps({
                    "id": cid, "ticker": tk, "form": form, "date": date,
                    "section": section, "text": piece, "n_tokens": n_tok(piece),
                }, ensure_ascii=False) + "\n")
                edges.append({"from": fid, "to": cid, "type": "HAS_CHUNK"})
                n_chunks += 1
        logger.info(f"[{tk}] {meta['sector']} — {len(list((DATA/tk).glob('*.txt')))} filings")

    chunk_f.close()
    json.dump({
        "companies": list(companies.values()),
        "sectors": sorted(sectors),
        "filings": list(filings.values()),
        "edges": edges,
    }, open(OUT / "graph.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    logger.info(f"[DONE] {len(companies)} companies, {len(sectors)} sectors, "
                f"{len(filings)} filings, {n_chunks} chunks, {len(edges)} edges")
    logger.info(f"  -> {OUT/'chunks.jsonl'}")
    logger.info(f"  -> {OUT/'graph.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Only first N companies (testing)")
    args = ap.parse_args()
    main(args.limit)
