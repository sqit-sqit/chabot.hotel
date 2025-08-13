
from __future__ import annotations
import re
import numpy as np
import tiktoken

def clean_text(text: str) -> str:
    # Basic normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def num_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    # Tokenizer choice doesn't have to match chat model exactly; rough sizing is fine
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
