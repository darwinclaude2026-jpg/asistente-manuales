import os
import glob
import time
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types

st.set_page_config(page_title="Asistente de Manuales Técnicos", page_icon="🔧", layout="centered")

GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
MANUALS_DIR = "manuales"
CHUNK_SIZE = 1500       # caracteres aprox. por fragmento
CHUNK_OVERLAP = 200


@st.cache_resource(show_spinner=False)
def get_client():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Falta configurar GEMINI_API_KEY en Settings → Secrets de Streamlit Cloud.")
        st.stop()
    return genai.Client(api_key=api_key)


def extract_chunks_from_pdfs(folder):
    chunks = []
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    for path in pdf_paths:
        filename = os.path.basename(path)
        try:
            reader = PdfReader(path)
        except Exception as e:
            st.warning(f"No se pudo leer {filename}: {e}")
            continue
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                piece = text[start:end].strip()
                if piece:
                    chunks.append({"text": piece, "source": filename, "page": page_num})
                start = end - CHUNK_OVERLAP
    return chunks


def embed_texts(client, texts, task_type, show_progress=False):
    vectors = []
    batch_size = 50  # menos llamadas = menos probabilidad de 429
    num_batches = (len(texts) + batch_size - 1) // batch_size
    progress = st.progress(0.0) if show_progress and num_batches > 1 else None

    for batch_num, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                vectors.extend([e.values for e in result.embeddings])
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = min(10 * (attempt + 1), 60)  # espera creciente: 10, 20, 30... hasta 60s
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(
                "Se alcanzó el límite gratuito de la API varias veces seguidas. "
                "Espera unos minutos (o hasta mañana si es el límite diario) y recarga la app."
            )
        # pausa entre lotes para respetar el límite de solicitudes por minuto del tier gratuito
        if i + batch_size < len(texts):
            time.sleep(4)
        if progress:
            progress.progress((batch_num + 1) / num_batches)

    if progress:
        progress.empty()
    return np.array(vectors, dtype=np.float32)


@st.cache_resource(show_spinner="Leyendo e indexando los manuales, esto puede tardar unos minutos la primera vez...")
def build_index():
    client = get_client()
    chunks = extract_chunks_from_pdfs(MANUALS_DIR)
    if not chunks:
        return client, [], None
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(client, texts, task_type="RETRIEVAL_DOCUMENT", show_progress=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms
    return client, chunks, vectors


def search(client, chunks, vectors, query, top_k=5):
    q_vec = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
    scores = vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def build_prompt(query, relevant_chunks):
    context = "\n\n".join(
        f"[Manual: {c['source']} - página {c['page']}]\n{c['text']}"
        for c in relevant_chunks
    )
    return f"""Eres un asistente técnico para mecánicos y operadores de maquinaria agrícola.
Responde la pregunta del técnico USANDO SOLO la información de los siguientes fragmentos de manuales.
Si la respuesta no está en los fragmentos, dilo claramente y no inventes.
Responde en español, de forma clara y práctica, e indica de qué manual y página sacaste la información.

FRAGMENTOS DE MANUALES:
{context}

PREGUNTA DEL TÉCNICO:
{query}

RESPUESTA:"""


st.title("🔧 Asistente de Manuales Técnicos")
st.caption("Consulta sobre mantenimiento de maquinaria agrícola, basado en tus manuales.")

client, chunks, vectors = build_index()

if not chunks:
    st.warning(
        "No se encontraron manuales PDF en la carpeta 'manuales/'. "
        "Sube tus PDFs a esa carpeta en el repositorio de GitHub y vuelve a desplegar."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Escribe tu consulta sobre mantenimiento...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los manuales..."):
            relevant = search(client, chunks, vectors, query)
            prompt = build_prompt(query, relevant)
            response = client.models.generate_content(model=GEN_MODEL, contents=prompt)
            answer = response.text
        st.markdown(answer)
        with st.expander("Ver fragmentos de manuales usados"):
            for c in relevant:
                st.markdown(f"**{c['source']} - página {c['page']}**")
                st.text(c["text"][:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
