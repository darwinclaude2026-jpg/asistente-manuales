import streamlit as st
import fitz  # PyMuPDF
import os
import glob
import numpy as np
from google import genai
from google.genai import types

st.set_page_config(page_title="Asistente de Manuales Técnicos", page_icon="🔧", layout="wide")

# --- Configuración ---
MANUALES_DIR = "manuales"
CHUNK_SIZE = 800       # caracteres aproximados por fragmento
CHUNK_OVERLAP = 150    # solape entre fragmentos para no perder contexto
TOP_K = 4              # cantidad de fragmentos que se pasan al modelo
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-3.5-flash"
EMBED_BATCH = 90       # fragmentos por lote al pedir embeddings

# --- API Key de Gemini (se configura en "Secrets" de Streamlit Cloud) ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error(
        "Falta configurar la API Key de Gemini. "
        "Ve a la configuración de la app en Streamlit Cloud > Settings > Secrets "
        "y agrega: GEMINI_API_KEY = \"tu_clave_aqui\""
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


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


def generar_embeddings(textos, task_type):
    """Pide embeddings a la API de Gemini en lotes."""
    vectores = []
    for i in range(0, len(textos), EMBED_BATCH):
        lote = textos[i:i + EMBED_BATCH]
        resultado = client.models.embed_content(
            model=EMBED_MODEL,
            contents=lote,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectores.extend([e.values for e in resultado.embeddings])
    arr = np.array(vectores)
    normas = np.linalg.norm(arr, axis=1, keepdims=True)
    normas[normas == 0] = 1
    return arr / normas


@st.cache_resource(show_spinner="Procesando manuales, esto puede tardar un minuto...")
def construir_indice():
    rutas = glob.glob(os.path.join(MANUALES_DIR, "*.pdf")) + glob.glob(os.path.join(MANUALES_DIR, "*.txt"))

    if not rutas:
        return None, []

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
    embeddings = generar_embeddings(textos, task_type="RETRIEVAL_DOCUMENT")
    return embeddings, todos_fragmentos


def buscar_fragmentos_relevantes(pregunta, embeddings, fragmentos, k=TOP_K):
    pregunta_emb = generar_embeddings([pregunta], task_type="RETRIEVAL_QUERY")[0]
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

    respuesta = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return respuesta.text


# --- Interfaz ---
st.title("🔧 Asistente de Manuales Técnicos")
st.caption("Consulta sobre mantenimiento de maquinaria agrícola, basado en tus manuales.")

embeddings, fragmentos = construir_indice()

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
            relevantes = buscar_fragmentos_relevantes(pregunta, embeddings, fragmentos)
            respuesta = generar_respuesta(pregunta, relevantes)
            st.markdown(respuesta)
            with st.expander("📄 Fuentes consultadas"):
                for f in relevantes:
                    st.caption(f"{f['manual']} - Página {f['pagina']}")

    st.session_state.historial.append(("assistant", respuesta))
