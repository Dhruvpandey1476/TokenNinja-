# Deploying TokenNinja on Railway

Two services from this one repo: **backend** (FastAPI) and **frontend** (static Vite build). Both use Docker.

## 1. Create the project
1. Push the repo to GitHub (done: `Dhruvpandey1476/TokenNinja-`).
2. Railway → **New Project → Deploy from GitHub repo** → pick the repo.
   Railway pulls Git LFS automatically, so `chunk_emb.npz` comes through.

## 2. Backend service
- **Settings → Root Directory:** `/`  ·  **Builder:** Dockerfile (`./Dockerfile`, auto-detected)
- **Settings → Healthcheck Path:** `/health`
- **Settings → Resources:** give it **≥ 2 GB RAM** (it loads PyTorch + the embedding model + the 161 MB embedding cache).
- **Variables** (from your `.env`):
  ```
  TIGERGRAPH_HOST=tg-xxxx.tg-xxxx.i.tgcloud.io
  TIGERGRAPH_GRAPH=GraphRAGDemo
  TIGERGRAPH_USERNAME=...
  TIGERGRAPH_PASSWORD=...
  TIGERGRAPH_SECRET=...
  TIGERGRAPH_PORT=443
  GEMINI_API_KEY=...            # or GOOGLE_API_KEY
  GEMINI_MODEL=gemini-2.5-flash # optional
  MAX_OUTPUT_TOKENS=1000        # optional
  ```
- **Settings → Networking → Generate Domain.** Copy the URL, e.g. `https://tokenninja-backend.up.railway.app`.
- Confirm it's live: open `<backend-url>/health` → `{"status":"ok",...}`.

## 3. Frontend service
- In the same project: **New → GitHub Repo → same repo** (adds a second service).
- **Settings → Root Directory:** `frontend`  ·  **Builder:** Dockerfile (`frontend/Dockerfile`)
- **Variables:**
  ```
  VITE_API_URL=https://tokenninja-backend.up.railway.app   # ← the backend URL from step 2
  ```
  (Vite inlines this at **build** time — Railway passes it as the `VITE_API_URL` build arg.)
- **Settings → Networking → Generate Domain.** This is your **app URL** to open/share.
- If you set `VITE_API_URL` after the first build, hit **Redeploy** so it's baked in.

## 4. Go live
1. **Resume the TigerGraph Savanna workspace** (queries hang if it's suspended).
2. Open the frontend URL → run a question. First query is a touch slow (workspace wake + first query embed), then ~2–4 s.

## Notes
- **CORS** is open on the backend (`allow_origins=["*"]`), so the frontend can call it cross-origin.
- **LFS fallback:** if the embedding cache isn't materialized, the app still works — it just re-embeds chunks at query time (~700 ms slower). To slim the image, delete the `chunk_emb.npz` COPY line in `Dockerfile` and add it to `.dockerignore`.
- **Build time:** the backend image is ~2.5 GB (PyTorch + model) — first build takes a few minutes; later deploys are cached.
- **Cost:** the backend needs real RAM; the free trial may OOM. Hobby plan (or ≥2 GB) recommended.
