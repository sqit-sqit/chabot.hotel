
from __future__ import annotations
import numpy as np
from .index import FaissIndex, load_metadata
from .utils import l2_normalize

def retrieve(index: FaissIndex, query_vec: np.ndarray, k: int, metadata: list[dict]):
    # query_vec: shape (dim,) -> reshape to (1, dim)
    q = query_vec.reshape(1, -1)
    q = l2_normalize(q)
    D, I = index.search(q, k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        m = metadata[idx]
        hits.append({
            "score": float(score),
            "text": m["text"],
            "metadata": m["metadata"],
        })
    return hits
