import os

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set – add via Hugging Face Space Settings → Secrets")