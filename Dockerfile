# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System deps (git not needed at runtime, but keeps image lean) ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies first (layer cache) ───────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir groq

# ── Copy project source ────────────────────────────────────────────────────────
COPY . .

# ── Install pymlens package itself ─────────────────────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Create persistent data directory for SQLite DB ────────────────────────────
RUN mkdir -p /root/.pymlens

# ── Expose Streamlit default port ─────────────────────────────────────────────
EXPOSE 8501

# ── Copy and permission the entrypoint script ─────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
