# Asistente de Manuales Técnicos — Guía de instalación (sin programar)

Este chat responde preguntas de mantenimiento de maquinaria agrícola basándose
en los manuales PDF que tú subas. Sigue estos pasos en orden. No necesitas
escribir ni una línea de código, solo crear cuentas gratis y subir archivos.

---

## Paso 1: Crear cuenta en GitHub (gratis)

1. Ve a https://github.com y crea una cuenta (si no tienes una).
2. Haz clic en el botón **"+"** (arriba a la derecha) → **"New repository"**.
3. Ponle un nombre, por ejemplo `asistente-manuales`.
4. Marca la opción **"Public"**.
5. Haz clic en **"Create repository"**.

## Paso 2: Subir los archivos de este proyecto

1. Dentro de tu nuevo repositorio, haz clic en **"Add file" → "Upload files"**.
2. Arrastra estos 3 archivos que te entregué:
   - `app.py`
   - `requirements.txt`
   - `README.md`
3. Haz clic en **"Commit changes"**.
4. Ahora crea la carpeta de manuales: en **"Add file" → "Upload files"** nuevamente,
   arrastra tus manuales en PDF. En el nombre del archivo, escribe `manuales/` antes
   del nombre del PDF (por ejemplo `manuales/tractor-x200.pdf`) para que GitHub
   cree la carpeta automáticamente. También puedes arrastrar varios PDFs a la vez
   una vez que la carpeta `manuales` ya exista.
5. Haz clic en **"Commit changes"**.

## Paso 3: Obtener tu API Key gratuita de Gemini (Google)

1. Ve a https://aistudio.google.com/apikey
2. Inicia sesión con tu cuenta de Google.
3. Haz clic en **"Create API key"**.
4. Copia la clave que te aparece (es un texto largo). Guárdala, la vas a usar
   en el siguiente paso.

> Esta clave te permite usar Gemini gratis dentro de un límite generoso de
> consultas por día, suficiente para un equipo técnico normal.

## Paso 4: Publicar la app en Streamlit Community Cloud (gratis)

1. Ve a https://share.streamlit.io y entra con tu cuenta de GitHub.
2. Haz clic en **"New app"**.
3. Selecciona el repositorio `asistente-manuales` que creaste.
4. En "Main file path" escribe: `app.py`
5. Antes de darle a "Deploy", haz clic en **"Advanced settings"**.
6. En la sección **"Secrets"**, pega esto (reemplazando por tu clave real):
   ```
   GEMINI_API_KEY = "pega_aqui_tu_clave_de_google"
   ```
7. Haz clic en **"Deploy"**.
8. Espera 1-2 minutos mientras se instala todo. Al terminar, te dará una URL
   (algo como `https://asistente-manuales.streamlit.app`) que puedes compartir
   con tus técnicos.

---

## Cómo agregar o actualizar manuales después

Solo repite el Paso 2.4: sube el nuevo PDF a la carpeta `manuales/` en GitHub.
La app se actualiza sola en unos minutos.

## Límites de la versión gratuita

- **Streamlit Community Cloud**: gratis, pero la app "duerme" si nadie la usa
  por varios días (tarda ~30 segundos en despertar la próxima vez que alguien
  entra). Suficiente para uso normal de un equipo técnico.
- **Gemini API gratis**: tiene un límite de consultas por día (generoso, pero
  no ilimitado). Si tu equipo lo supera, se puede pasar a un plan pago barato
  o repartir en varias claves.
- Pensado para una cantidad moderada de manuales (hasta ~10-15 PDFs). Si más
  adelante necesitas manejar muchos más manuales o más usuarios simultáneos,
  conviene mover esto a un servidor propio con una base de datos vectorial
  persistente (Chroma/Qdrant) en vez de recalcular todo al iniciar.

## Si algo falla

- Revisa los "logs" en Streamlit Cloud (botón "Manage app" abajo a la derecha
  de tu app) — ahí aparece el error exacto en texto rojo.
- El error más común es olvidar poner la `GEMINI_API_KEY` en "Secrets".
