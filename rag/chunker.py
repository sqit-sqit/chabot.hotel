
from __future__ import annotations
from typing import Iterable
from .config import CHUNK_TOKENS, CHUNK_OVERLAP
from .utils import num_tokens

def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # Simple token-window chunking using tiktoken sizes
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for w in words:
        current.append(w)
        current_len = num_tokens(' '.join(current))
        if current_len >= chunk_tokens:
            chunk = ' '.join(current)
            chunks.append(chunk)
            # overlap
            if overlap > 0:
                # keep last words to approximate overlap by tokens
                keep = max(1, int(len(current) * overlap / max(current_len, 1)))
                current = current[-keep:]
            else:
                current = []
    if current:
        chunks.append(' '.join(current))
    return chunks

def chunk_documents(docs: Iterable[dict]) -> list[dict]:
    out = []
    for d in docs:
        chs = chunk_text(d["text"])
        for i, ch in enumerate(chs):
            out.append({
                "text": ch,
                "metadata": {
                    "source": d.get("source"),
                    "title": d.get("title"),
                    "url": d.get("url"),
                    "chunk_id": i,
                }
            })
    return out
