# Alerta temprana de crecidas del río Paraná en Rosario

App web que usa un árbol de clasificación para estimar, con 7 días de anticipación, si el río Paraná en Rosario estará en nivel **Normal**, **Alerta** o **Evacuación** (umbrales oficiales del INA: 5,00 m y 5,25 m).

**Materia:** Modelizado de Sistemas de IA (PPII) – Unidad 3 – IFTS N° 18
**Grupo:** Leanderson Carvalho · Stefania Silva · Guillermo Seron · Richard Quintero · Gonzalo Yahuita

> ⚠️ Prototipo académico. No reemplaza las alertas oficiales del INA, el SMN ni Defensa Civil.

## Archivos

| Archivo | Para qué sirve |
|---|---|
| `app.py` | El código de la app |
| `requirements.txt` | Las librerías que necesita la app |
| `.streamlit/config.toml` | Colores de la app |
| `modelo_simple_rosario.joblib` | El modelo entrenado en Colab (**hay que agregarlo**, ver paso 1) |

Si el archivo `.joblib` no está, la app entrena el mismo árbol sola al iniciar, descargando los datos del INA desde 1980. Funciona igual, pero el primer arranque tarda unos minutos.

## Paso 1 · Descargar el modelo de Colab
1. Abrir `Notebook_Version_Simple_Parana.ipynb` en Colab y ejecutarlo completo.
2. En el panel de archivos (ícono de carpeta, a la izquierda), buscar `modelo_simple_rosario.joblib`.
3. Clic derecho → **Descargar**.
4. Copiarlo en esta carpeta, junto a `app.py`.

## Paso 2 · Subir la carpeta a GitHub
1. En github.com, crear un repositorio **público** (por ejemplo `alerta-parana`).
2. **Add file → Upload files** y subir `app.py`, `requirements.txt`, `README.md` y `modelo_simple_rosario.joblib`.
3. La carpeta `.streamlit` empieza con punto y a veces no se sube arrastrando. Si pasa, usar **Add file → Create new file**, escribir como nombre `.streamlit/config.toml` y pegar su contenido. Es opcional: sin ese archivo la app funciona con los colores por defecto.

## Paso 3 · Publicar en Streamlit Community Cloud
1. Entrar a share.streamlit.io e iniciar sesión con la cuenta de GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Elegir el repositorio, la rama `main` y el archivo `app.py`.
4. En **Advanced settings**, elegir **Python 3.11** o **3.12**.
5. Elegir la dirección de la app (por ejemplo `alerta-parana`) y hacer clic en **Deploy**.

En unos minutos la app queda en una dirección del tipo `https://alerta-parana.streamlit.app`.

## Paso 4 · Probar
- Debe mostrar el semáforo con la fecha del último dato del INA.
- Si aparece "No se pudieron descargar los datos del INA", el servidor del INA no respondió desde la nube: probar más tarde con el botón **Actualizar datos del INA**.

## Probar en la computadora (opcional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Tener en cuenta
- `scikit-learn` está fijado en **1.6.1**, la misma versión de Colab con la que se entrenó el modelo. Si se cambia, el `.joblib` puede no cargar.
- En el plan gratuito, la app **se duerme después de 12 horas sin visitas**. Abrirla unos minutos antes de presentar.
- Los datos del INA se guardan en memoria durante 1 hora; el botón **Actualizar datos del INA** fuerza una descarga nueva.
