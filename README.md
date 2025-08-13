
# RAG + FAISS (lokalnie) – szkielet pod DigitalOcean App Platform

Ten projekt to lekki szkielet **RAG z embeddingami i lokalnym FAISS**, z prostą aplikacją **Streamlit**.
Zaprojektowany tak, by działał lokalnie i na **DigitalOcean App Platform** z **Persistent Storage**.

## Szybki start (local)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# wpisz swój OPENAI_API_KEY w .env

python scripts/ingest.py --rebuild    # zbuduje indeks z ./data i ./data/urls.txt (jeśli istnieje)
streamlit run app_streamlit.py
```

## DigitalOcean App Platform (skrót)

1. W repo ustaw pliki tak jak tutaj, wypchnij na GitHub.
2. W App Platform:
   - **Source**: repozytorium.
   - **Service**: Web Service (Dockerfile).
   - **Environment Variables**: ustaw `OPENAI_API_KEY` (Required), `INDEX_DIR=/app/storage/index`.
   - **Persistent Storage**: dodaj wolumen, np. 1–5 GB, **Mount Path**: `/app/storage`.
   - **Run Command** (jeśli nadpisujesz): `bash -lc "streamlit run app_streamlit.py --server.port $PORT --server.address 0.0.0.0"`
3. Po uruchomieniu wejdź w aplikację i w panelu bocznym kliknij **Rebuild index** (lub uruchom `python scripts/ingest.py --rebuild` w konsoli DO).

## Struktura

```
.
├─ rag/                # moduł z logiką RAG
│  ├─ config.py
│  ├─ loader.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ index.py
│  ├─ retriever.py
│  ├─ generator.py
│  └─ utils.py
├─ scripts/
│  ├─ ingest.py        # buduje indeks (FAISS + metadata)
│  └─ query.py         # szybki test zapytania z terminala
├─ data/
│  └─ sample/docs/     # przykładowe dokumenty
├─ storage/            # (gitignored) – tu ląduje indeks i metadane
├─ app_streamlit.py    # prosty UI do zapytań i budowy indeksu
├─ requirements.txt
├─ Dockerfile
├─ .env.example
└─ README.md
```

## Notatki projektowe

- **FAISS** używa kosinusowej „podobieństwa” przez normalizację wektorów i `IndexFlatIP`.
- Metadane (źródło, tytuł, url) trzymane są w pliku `metadata.pkl` obok indeksu.
- **Chunking** sterowany `CHUNK_TOKENS` i `CHUNK_OVERLAP` (domyślnie 800/120).
- Prosty **scraper** HTML wyciąga tekst (BeautifulSoup), z opcją `ALLOW_INSECURE_SSL` (ostrożnie!).
- Generowanie odpowiedzi używa `CHAT_MODEL` z kontekstem top-K chunków.

## TODO (opcjonalne)

- Dodać **hybrydowe wyszukiwanie** (BM25 + FAISS) i **reranking**.
- Dodać **FastAPI** dla endpointów /query /ingest.
- Obsługa dokumentów PDF/Docx (np. `pypdf`, `docx2txt`).

Powodzenia! 🚀
