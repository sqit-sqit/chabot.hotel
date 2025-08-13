
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="RAG + FAISS (lokalnie)", page_icon="🔎", layout="wide")

from rag.config import INDEX_DIR, TOP_K, CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY
from rag.index import build_or_load, index_exists, load_metadata
from rag.embedder import embed_texts
from rag.retriever import retrieve
from rag.generator import answer
from rag.loader import load_local_docs, load_urls_file
from rag.chunker import chunk_documents
from rag.utils import l2_normalize

def ensure_api_key():
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key")
    if not key:
        st.info("Dodaj swój klucz API OpenAI, aby korzystać z aplikacji.")
        st.session_state["openai_api_key"] = st.text_input("OPENAI_API_KEY", type="password")
        if st.session_state["openai_api_key"]:
            os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
            st.rerun()
    return os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")
    st.write(f"**INDEX_DIR**: `{INDEX_DIR}`")
    st.write(f"**EMBED_MODEL**: `{EMBED_MODEL}`")
    st.write(f"**CHAT_MODEL**: `{CHAT_MODEL}`")
    st.divider()

    if st.button("🔁 Rebuild index"):
        with st.spinner("Buduję indeks..."):
            # simple inline ingest (to avoid separate script on DO)
            docs = []
            docs += load_local_docs("data")
            docs += load_urls_file("data/urls.txt")
            if not docs:
                st.error("Brak dokumentów. Dodaj pliki do ./data lub URL-e do data/urls.txt")
            else:
                chunks = chunk_documents(docs)
                texts = [c["text"] for c in chunks]
                vecs = l2_normalize(embed_texts(texts))

                from rag.index import FaissIndex, save_metadata
                index = FaissIndex(dim=vecs.shape[1])
                index.add(vecs)
                index.save()
                metadata = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
                save_metadata(metadata)
                st.success("Indeks przebudowany ✅")

    st.divider()
    top_k = st.slider("TOP_K", min_value=3, max_value=20, value=TOP_K, step=1)

st.title("🔎 RAG + FAISS (lokalnie)")
st.caption("Prosty szkielet do wdrożenia na DigitalOcean App Platform (z Persistent Storage).")

ensure_api_key()

if not index_exists():
    st.warning("Indeks jeszcze nie istnieje. Zbuduj go przyciskiem po lewej.")
else:
    # Load index & metadata
    index, meta = build_or_load(dim=1536, rebuild=False)  # dim nieużywany przy load
    if not meta:
        meta = load_metadata()

    q = st.text_input("Twoje pytanie:", placeholder="Np. Jakie pakiety są dostępne w SPA?")
    if q:
        with st.spinner("Wyszukuję i generuję odpowiedź..."):
            qv = embed_texts([q])[0]
            hits = retrieve(index, qv, top_k, meta)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Kontekst (top-K)")
                for i, h in enumerate(hits, start=1):
                    m = h["metadata"]
                    label = f"{i}. {m.get('title') or m.get('source')}"
                    if m.get("url"):
                        label += f" — {m['url']}"
                    st.markdown(f"**{label}**  (score={h['score']:.3f})")
                    st.write(h["text"][:800] + ("…" if len(h["text"]) > 800 else ""))
                    st.divider()
            with col2:
                st.subheader("Odpowiedź")
                ans = answer(q, hits)
                st.write(ans)



# --- Diagnostyka URL (lokalnie) ---
import streamlit as st
with st.expander("🔎 Diagnostyka URL (podgląd ekstrakcji)"):
    test_url = st.text_input("Wklej URL do testu", "")
    if st.button("Pobierz i pokaż tekst"):
        from rag.loader import fetch_url
        if test_url:
            txt = fetch_url(test_url)
            if not txt:
                st.error("❌ Brak tekstu (SSL/403/JS-only?)")
            else:
                st.success(f"✅ Pobrano {len(txt)} znaków.")
                st.write(txt[:3000] + ("…" if len(txt) > 3000 else ""))
