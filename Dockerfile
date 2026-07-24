# ── Backend: FastAPI 3-pipeline comparison API ───────────────────────────────
# Deploy target: Railway (service root = repo root, Dockerfile = ./Dockerfile)
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first so sentence-transformers doesn't pull the ~2GB CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake the embedding model into the image → no HuggingFace download at boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# App code + the small data files the pipelines read at runtime.
COPY backend/ backend/
COPY evaluation/ evaluation/
COPY data/round3/graph.json data/round3/graph.json
# Precomputed chunk embeddings (Git LFS). Keeps GraphRAG's optimizer near-instant.
# If LFS isn't materialized the app falls back to runtime embedding (just slower),
# so a missing/pointer file won't break the build. Delete this COPY to slim ~161MB.
COPY data/round3/chunk_emb.npz data/round3/chunk_emb.npz

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Railway injects $PORT; default to 8000 locally.
CMD uvicorn backend.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
