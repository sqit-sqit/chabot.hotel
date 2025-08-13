
from __future__ import annotations
from typing import List
from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM = """
Jesteś pomocnym asystentem RAG. Odpowiadasz na pytanie wyłącznie na podstawie dostarczonego kontekstu.
Jeśli brakuje informacji, powiedz, że ich nie masz.
Na końcu podaj listę cytowanych źródeł z tytułem i (jeśli jest) adresem URL.
"""

def format_context(hits: List[dict]) -> str:
    blocks = []
    for h in hits:
        meta = h.get("metadata", {}) or {}
        src = meta.get("title") or meta.get("source") or "nieznane źródło"
        url = meta.get("url")
        head = f"Źródło: {src}" + (f" ({url})" if url else "")
        blocks.append(f"{head}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)

def answer(query: str, hits: List[dict]) -> str:
    context = format_context(hits)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Kontekst:\n{context}\n\nPytanie: {query}"},
    ]
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
