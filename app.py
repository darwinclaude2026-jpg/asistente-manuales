import os
import glob
import json
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
INDEX_FILE = "manuales_index.npz"
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

    # --- Archivos PDF ---
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
                    chunks.append({"text": piece, "source": filename, "location": f"página {page_num}"})
                start = end - CHUNK_OVERLAP

    # --- Archivos TXT (útil cuando el PDF original es muy pesado para GitHub) ---
    txt_paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    for path in txt_paths:
        filename = os.path.basename(path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            st.warning(f"No se pudo leer {filename}: {e}")
            continue
        start = 0
        section = 1
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "source": filename, "location": f"sección {section}"})
                section += 1
            start = end - CHUNK_OVERLAP

    return chunks


def embed_texts(client, texts, task_type, show_progress=False, on_batch_done=None):
    """Devuelve (vectores_calculados, se_completo_todo).
    Si choca con el límite de la API, NO lanza error: devuelve lo que alcanzó a procesar.
    on_batch_done(vectores_hasta_ahora) se llama después de cada lote exitoso, para poder
    guardar el progreso parcial en disco."""
    vectors = []
    batch_size = 50  # menos llamadas = menos probabilidad de 429
    num_batches = (len(texts) + batch_size - 1) // batch_size
    progress = st.progress(0.0) if show_progress and num_batches > 1 else None

    for batch_num, i in enumerate(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        max_attempts = 3  # reintentos cortos: si es límite DIARIO, esperar más no sirve de nada
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
                    time.sleep(15 * (attempt + 1))  # 15s, 30s, 45s
                else:
                    raise
        else:
            # Se agotó el límite: paramos aquí, pero guardamos lo ya avanzado
            if progress:
                progress.empty()
            if on_batch_done:
                on_batch_done(vectors)
            return vectors, False

        if on_batch_done:
            on_batch_done(vectors)
        if i + batch_size < len(texts):
            time.sleep(4)  # pausa entre lotes para respetar el límite por minuto
        if progress:
            progress.progress((batch_num + 1) / num_batches)

    if progress:
        progress.empty()
    return vectors, True


def save_index(chunks, vectors):
    try:
        np.savez(INDEX_FILE, vectors=vectors, chunks_json=json.dumps(chunks))
    except Exception as e:
        st.warning(f"No se pudo guardar el índice en disco: {e}")


def load_index():
    if not os.path.exists(INDEX_FILE):
        return None, None
    try:
        data = np.load(INDEX_FILE, allow_pickle=True)
        chunks = json.loads(str(data["chunks_json"]))
        vectors = data["vectors"]
        return chunks, vectors
    except Exception:
        return None, None


@st.cache_resource(show_spinner="Leyendo e indexando los manuales, esto puede tardar unos minutos la primera vez...")
def build_index():
    client = get_client()
    chunks = extract_chunks_from_pdfs(MANUALS_DIR)
    if not chunks:
        return client, [], None, "sin_manuales"

    cached_chunks, cached_vectors = load_index()

    # ¿El progreso guardado corresponde al mismo inicio de estos mismos manuales?
    resume_from = 0
    existing_vectors = None
    if cached_chunks and cached_vectors is not None and len(cached_vectors) > 0:
        n = min(len(cached_chunks), len(chunks))
        coincide = all(
            cached_chunks[i]["text"] == chunks[i]["text"] and cached_chunks[i]["source"] == chunks[i]["source"]
            for i in range(n)
        )
        if coincide:
            resume_from = n
            existing_vectors = np.array(cached_vectors[:n], dtype=np.float32)

    if resume_from >= len(chunks):
        return client, chunks, existing_vectors, "completo_desde_cache"

    texts_pendientes = [c["text"] for c in chunks[resume_from:]]

    def guardar_progreso(nuevos_vectores):
        if len(nuevos_vectores) == 0 and existing_vectors is None:
            return
        combinados = np.array(nuevos_vectores, dtype=np.float32)
        if existing_vectors is not None and len(existing_vectors) > 0:
            combinados = np.vstack([existing_vectors, combinados]) if len(combinados) else existing_vectors
        save_index(chunks[:resume_from + len(nuevos_vectores)], combinados)

    nuevos_vectores, completo = embed_texts(
        client, texts_pendientes, task_type="RETRIEVAL_DOCUMENT",
        show_progress=True, on_batch_done=guardar_progreso,
    )

    total_vectores = np.array(nuevos_vectores, dtype=np.float32)
    if existing_vectors is not None and len(existing_vectors) > 0:
        total_vectores = np.vstack([existing_vectors, total_vectores]) if len(total_vectores) else existing_vectors

    if len(total_vectores) == 0:
        return client, [], None, "limite_alcanzado_sin_avance"

    norms = np.linalg.norm(total_vectores, axis=1, keepdims=True)
    norms[norms == 0] = 1
    total_vectores_norm = total_vectores / norms
    chunks_procesados = chunks[:len(total_vectores)]
    save_index(chunks_procesados, total_vectores_norm)

    if not completo:
        return client, chunks_procesados, total_vectores_norm, "limite_alcanzado_parcial"
    return client, chunks, total_vectores_norm, "completo_recien_calculado"


def search(client, chunks, vectors, query, top_k=5):
    q_vecs, _ = embed_texts(client, [query], task_type="RETRIEVAL_QUERY")
    q_vec = np.array(q_vecs[0], dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
    scores = vectors @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def build_prompt(query, relevant_chunks):
    context = "\n\n".join(
        f"[Manual: {c['source']} - {c['location']}]\n{c['text']}"
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

client, chunks, vectors, estado = build_index()

if estado == "sin_manuales":
    st.warning(
        "No se encontraron manuales (PDF o TXT) en la carpeta 'manuales/'. "
        "Sube tus archivos a esa carpeta en el repositorio de GitHub y vuelve a desplegar."
    )
    st.stop()

if estado == "limite_alcanzado_sin_avance":
    st.error(
        "Se alcanzó el límite gratuito de la API antes de poder procesar ningún fragmento. "
        "Espera unos minutos (o hasta la madrugada, si es el límite diario) y recarga la página."
    )
    st.stop()

if estado in ("completo_recien_calculado", "limite_alcanzado_parcial") and os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        index_bytes = f.read()
    if estado == "completo_recien_calculado":
        st.info(
            "✅ Índice generado por completo. Para que no se vuelva a gastar cuota de la API en cada "
            "reinicio, descarga este archivo y súbelo a la raíz de tu repositorio de GitHub (junto a app.py)."
        )
    else:
        st.warning(
            f"⚠️ Se alcanzó el límite gratuito de la API a mitad de camino. Se procesaron "
            f"{len(chunks)} fragmentos y quedaron pendientes los demás. Descarga este archivo parcial "
            f"y súbelo a GitHub como `manuales_index.npz` — la próxima vez (cuando se libere la cuota) "
            f"la app continuará automáticamente desde donde quedó, en vez de repetir todo."
        )
    st.download_button(
        "⬇️ Descargar manuales_index.npz",
        data=index_bytes,
        file_name="manuales_index.npz",
        mime="application/octet-stream",
    )

if estado == "limite_alcanzado_parcial":
    st.info("Puedes seguir usando el chat mientras tanto — solo que buscará únicamente en los fragmentos ya procesados.")

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
                st.markdown(f"**{c['source']} - {c['location']}**")
                st.text(c["text"][:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
