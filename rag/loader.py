
from __future__ import annotations
import os, glob, requests
from bs4 import BeautifulSoup
from .config import REQUEST_TIMEOUT, ALLOW_INSECURE_SSL
from .utils import clean_text

def load_local_docs(root: str = "data") -> list[dict]:
    docs = []
    # Load .txt and .md
    for pattern in ["**/*.txt", "**/*.md"]:
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                docs.append({
                    "text": clean_text(text),
                    "source": path,
                    "title": os.path.basename(path),
                    "url": None,
                })
            except Exception as e:
                print(f"[load_local_docs] Skip {path}: {e}")
    return docs

def fetch_url(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, verify=not ALLOW_INSECURE_SSL, headers={
            "User-Agent": "RAG-FAISS/1.0 (+https://example.com)"
        })
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style/nav
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return clean_text(text)
    except Exception as e:
        print(f"[fetch_url] {url} -> {e}")
        return None

def load_urls_file(path: str = "data/urls.txt") -> list[dict]:
    if not os.path.exists(path):
        return []
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            text = fetch_url(url)
            if text:
                docs.append({
                    "text": text,
                    "source": "url",
                    "title": url,
                    "url": url,
                })
    return docs
