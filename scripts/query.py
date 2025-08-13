
from __future__ import annotations
import sys
from dotenv import load_dotenv
load_dotenv()

from rag.embedder import embed_texts
from rag.index import build_or_load, load_metadata
from rag.retriever import retrieve
from rag.generator import answer
from rag.config import TOP_K

def main(question: str):
    # Load index & metadata
    index, meta = build_or_load(dim=1536, rebuild=False)  # dim not used when loading existing
    if not meta:
        meta = load_metadata()
    # Embed query
    qv = embed_texts([question])[0]
    hits = retrieve(index, qv, TOP_K, meta)
    print("== Context ==")
    for h in hits:
        m = h["metadata"]
        print(f"- {m.get('title') or m.get('source')} (score={h['score']:.3f})")
    print("\n== Answer ==")
    print(answer(question, hits))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/query.py "Twoje pytanie"")
        sys.exit(1)
    main(sys.argv[1])
