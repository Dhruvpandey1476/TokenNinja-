# ── Frontend: build the Vite/React app, serve the static bundle ──────────────
# Deploy target: Railway (service root = frontend/, Dockerfile = frontend/Dockerfile)
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Bake the backend URL at build time (Vite inlines VITE_* at build).
# On Railway set VITE_API_URL (service variable) to the backend's public URL.
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Serve the static SPA.
FROM node:20-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=build /app/dist ./dist
EXPOSE 4173
CMD serve -s dist -l tcp://0.0.0.0:${PORT:-4173}
