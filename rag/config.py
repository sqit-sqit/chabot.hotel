
from __future__ import annotations
import os

def env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

# Required
OPENAI_API_KEY = env("OPENAI_API_KEY", None)

# Optional
INDEX_DIR      = os.getenv("INDEX_DIR", "./storage/index")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_TOKENS   = int(os.getenv("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "120"))

TOP_K          = int(os.getenv("TOP_K", "5"))
REQUEST_TIMEOUT= int(os.getenv("REQUEST_TIMEOUT", "15"))
ALLOW_INSECURE_SSL = os.getenv("ALLOW_INSECURE_SSL", "0") == "1"

# Ensure index dir exists
os.makedirs(INDEX_DIR, exist_ok=True)
