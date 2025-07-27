# backend/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
HF_TOKEN = os.getenv("HF_TOKEN")