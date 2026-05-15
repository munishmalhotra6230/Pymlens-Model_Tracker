#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────────────────────
# If a GROQ_API_KEY environment variable is passed, write it to secrets.toml
# so the Streamlit Critics page can use it via st.secrets['MY_API_KEY'].
# This avoids ever baking the real key into the image.
# ─────────────────────────────────────────────────────────────────────────────
if [ -n "$GROQ_API_KEY" ]; then
    mkdir -p /app/pymlens/.streamlit
    echo "MY_API_KEY = '${GROQ_API_KEY}'" > /app/pymlens/.streamlit/secrets.toml
    echo "✅ Groq API key configured."
else
    echo "⚠️  No GROQ_API_KEY set — Critics page will not work."
fi

echo "🚀 Starting PyMLens dashboard on port 8501..."

exec streamlit run /app/pymlens/dashboard.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
