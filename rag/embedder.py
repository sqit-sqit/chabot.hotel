
from __future__ import annotations
from typing import List
import numpy as np
from openai import OpenAI
from .config import OPENAI_API_KEY, EMBED_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI embeddings API returns lists-of-floats; convert to np.array
    # Batch in reasonable sizes to avoid payload limits
    BATCH = 256
    vectors = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)
        vs = [e.embedding for e in resp.data]
        vectors.extend(vs)
    return np.array(vectors, dtype="float32")
