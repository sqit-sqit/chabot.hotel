
from __future__ import annotations
import os, pickle
import faiss
import numpy as np
from .config import INDEX_DIR
from .utils import l2_normalize

INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH  = os.path.join(INDEX_DIR, "metadata.pkl")

class FaissIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # use cosine via normalized vectors
        self._normalized = True

    def add(self, vectors: np.ndarray):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)

    def search(self, vectors: np.ndarray, k: int):
        return self.index.search(vectors, k)

    def save(self, path: str = INDEX_PATH):
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str = INDEX_PATH) -> "FaissIndex":
        index = faiss.read_index(path)
        obj = cls(index.d)
        obj.index = index
        return obj

def save_metadata(meta: list[dict], path: str = META_PATH):
    with open(path, "wb") as f:
        pickle.dump(meta, f)

def load_metadata(path: str = META_PATH) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)

def index_exists() -> bool:
    return os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)

def build_or_load(dim: int, rebuild: bool = False):
    if not rebuild and index_exists():
        return FaissIndex.load(), load_metadata()
    os.makedirs(INDEX_DIR, exist_ok=True)
    # Caller must add vectors and save
    return FaissIndex(dim), []
