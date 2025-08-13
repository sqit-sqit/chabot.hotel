
from __future__ import annotations
import os, argparse
from dotenv import load_dotenv
load_dotenv()

from rag.config import INDEX_DIR, TOP_K
from rag.loader import load_local_docs, load_urls_file
from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.index import build_or_load, save_metadata
from rag.utils import l2_normalize

def main(rebuild: bool = False):
    print(f"[ingest] INDEX_DIR={INDEX_DIR}")
    # 1) Load docs
    docs = []
    docs += load_local_docs("data")
    docs += load_urls_file("data/urls.txt")
    if not docs:
        print("[ingest] Brak dokumentów. Dodaj pliki do ./data lub URL-e do data/urls.txt")
        return

    # 2) Chunk
    chunks = chunk_documents(docs)
    print(f"[ingest] Documents: {len(docs)}, Chunks: {len(chunks)}")

    # 3) Embed
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    vecs = l2_normalize(vecs)

    # 4) Build index
    index, meta = build_or_load(dim=vecs.shape[1], rebuild=True)  # force rebuild
    index.add(vecs)
    index.save()
    # Save metadata alongside
    metadata = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
    save_metadata(metadata)
    print(f"[ingest] ✅ Index built and saved at {INDEX_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
