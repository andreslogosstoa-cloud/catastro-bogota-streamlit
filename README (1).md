# Aplicación Streamlit — Catastro Bogotá 2023

## Ejecutar localmente
1. Instala Python 3.10 o superior.
2. Abre una terminal en esta carpeta.
3. Ejecuta:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Se abrirá la aplicación en `http://localhost:8501`.
5. En el panel lateral carga `avalcatmz2023.zip`.

## Publicar y obtener un enlace público
1. Crea un repositorio en GitHub.
2. Sube `app.py` y `requirements.txt` a la raíz del repositorio.
3. Entra a https://share.streamlit.io/ e inicia sesión con GitHub.
4. Selecciona **Create app**, el repositorio, la rama y `app.py`.
5. Pulsa **Deploy**. Streamlit generará una URL pública `https://....streamlit.app`.
6. Comparte esa URL. La aplicación solicita que cada visitante cargue el ZIP de datos.

## Nota académica
La app estima el avalúo mediano por manzana en Bogotá para la vigencia 2023. No es un avalúo oficial ni predice el valor de un predio individual.
