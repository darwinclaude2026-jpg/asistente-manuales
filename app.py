import streamlit as st
import fitz  # PyMuPDF
import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="Asistente de Manuales Técnicos", page_icon="🔧", layout="wide")

# --- Configuración ---
MANUALES_DIR = "manuales"
CHUNK_SIZE = 800       # caracteres aproximados por fragmento
CHUNK_OVERLAP = 150    # solape entre fragmentos para no perder contexto
TOP_K = 4              # cantidad de fragmentos que se pasan al modelo

# --- API Key de Gemini (se configura en "Secrets" de Streamlit Cloud) ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error(
        "Falta configurar la API Key de Gemini. "
        "Ve a la configuración de la app en Streamlit Cloud > Settings > Secrets "
        "y agrega: GEMINI_API_KEY = \"tu_clave_aqui\""
    )
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)


@st.cache_resource(show_spinner="Cargando modelo de búsqueda...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extraer_texto_pdf(ruta):
    doc = fitz.open(ruta)
    paginas = []
    for i, page in enumerate(doc):
        texto = page.get_text()
        if texto.strip():
            paginas.append((i + 1, texto))
    doc.close()
    return paginas


def extraer_texto_txt(ruta):
    """Lee archivos .txt exportados con marcadores '<<<PAGINA N>>>' entre páginas."""
    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()

    paginas = []
    bloques = contenido.split("<<<PAGINA ")
    for bloque in bloques:
        if not bloque.strip():
            continue
        try:
            num_str, resto = bloque.split(">>>", 1)
            num_pagina = int(num_str.strip())
        except ValueError:
            continue
        if resto.strip():
            paginas.append((num_pagina, resto))
    return paginas


def dividir_en_fragmentos(texto, tamano=CHUNK_SIZE, solape=CHUNK_OVERLAP):
    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano
        fragmentos.append(texto[inicio:fin])
        inicio += tamano - solape
    return fragmentos


@st.cache_resource(show_spinner="Procesando manuales, esto puede tardar un minuto...")
def construir_indice():
    embedder = load_embedder()
    rutas = glob.glob(os.path.join(MANUALES_DIR, "*.pdf")) + glob.glob(os.path.join(MANUALES_DIR, "*.txt"))

    if not rutas:
        return None, [], embedder

    todos_fragmentos = []  # lista de dicts: texto, manual, pagina
    for ruta in rutas:
        nombre_manual = os.path.basename(ruta)
        if ruta.lower().endswith(".pdf"):
            paginas = extraer_texto_pdf(ruta)
        else:
            paginas = extraer_texto_txt(ruta)
        for num_pagina, texto_pagina in paginas:
            for frag in dividir_en_fragmentos(texto_pagina):
                if frag.strip():
                    todos_fragmentos.append({
                        "texto": frag,
                        "manual": nombre_manual,
                        "pagina": num_pagina
                    })

    textos = [f["texto"] for f in todos_fragmentos]
    embeddings = embedder.encode(textos, show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings), todos_fragmentos, embedder


def buscar_fragmentos_relevantes(pregunta, embeddings, fragmentos, embedder, k=TOP_K):
    pregunta_emb = embedder.encode([pregunta], normalize_embeddings=True)[0]
    similitudes = embeddings @ pregunta_emb
    indices_top = np.argsort(similitudes)[::-1][:k]
    return [fragmentos[i] for i in indices_top]


def generar_respuesta(pregunta, contexto_fragmentos):
    contexto_texto = "\n\n".join(
        f"[Manual: {f['manual']} - Página {f['pagina']}]\n{f['texto']}"
        for f in contexto_fragmentos
    )
    prompt = f"""Eres un asistente técnico experto en mantenimiento de maquinaria agrícola.
Responde la pregunta del técnico usando ÚNICAMENTE la información del contexto de los manuales que te doy abajo.
Si la respuesta no está en el contexto, dilo claramente y no inventes información.
Cuando respondas, indica de qué manual y página sacaste la información.

CONTEXTO DE LOS MANUALES:
{contexto_texto}

PREGUNTA DEL TÉCNICO:
{pregunta}

RESPUESTA:"""

    modelo = genai.GenerativeModel("gemini-1.5-flash")
    respuesta = modelo.generate_content(prompt)
    return respuesta.text


# --- Interfaz ---
st.title("🔧 Asistente de Manuales Técnicos")
st.caption("Consulta sobre mantenimiento de maquinaria agrícola, basado en tus manuales.")

embeddings, fragmentos, embedder = construir_indice()

if embeddings is None:
    st.warning(
        f"No se encontraron manuales (PDF o TXT) en la carpeta '{MANUALES_DIR}/'. "
        "Sube tus manuales a esa carpeta en el repositorio de GitHub."
    )
    st.stop()

st.success(f"✅ {len(set(f['manual'] for f in fragmentos))} manual(es) cargado(s). Listo para consultar.")

if "historial" not in st.session_state:
    st.session_state.historial = []

for rol, mensaje in st.session_state.historial:
    with st.chat_message(rol):
        st.markdown(mensaje)

pregunta = st.chat_input("Escribe tu consulta sobre mantenimiento...")

if pregunta:
    st.session_state.historial.append(("user", pregunta))
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los manuales..."):
            relevantes = buscar_fragmentos_relevantes(pregunta, embeddings, fragmentos, embedder)
            respuesta = generar_respuesta(pregunta, relevantes)
            st.markdown(respuesta)
            with st.expander("📄 Fuentes consultadas"):
                for f in relevantes:
                    st.caption(f"{f['manual']} - Página {f['pagina']}")

    st.session_state.historial.append(("assistant", respuesta))
