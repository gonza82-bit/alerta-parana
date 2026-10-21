"""
Alerta temprana de crecidas del río Paraná en Rosario
App web en Streamlit que usa el árbol de clasificación (versión simple, 5 variables).

Materia: Modelizado de Sistemas de IA (PPII) – Unidad 3 – IFTS N° 18
Grupo: Leanderson Carvalho, Stefania Silva, Guillermo Seron, Richard Quintero, Gonzalo Yahuita

Prototipo académico: no reemplaza las alertas oficiales del INA, el SMN ni Defensa Civil.
"""

import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# 1. Configuración (los mismos valores que el notebook)
# --------------------------------------------------------------------------
st.set_page_config(page_title="Alerta Paraná – Rosario", page_icon="🌊", layout="wide")

BASE_INA = "https://alerta.ina.gob.ar/a5/getObservaciones"
SERIES_INA = {"Rosario": 34, "Corrientes": 19, "Goya": 23, "Reconquista": 24}
ALERTA, EVACUACION = 5.00, 5.25          # umbrales oficiales del INA en Rosario (m)
HORIZONTE = 7                            # días hacia adelante
DIAS_GRAFICO = 120                       # días que muestra el gráfico

VARIABLES = ["altura_hoy", "altura_corrientes", "altura_goya",
             "altura_reconquista", "var_reconquista_7d"]
NOMBRES = {
    "altura_hoy": "la altura del río en Rosario",
    "altura_corrientes": "la altura en Corrientes",
    "altura_goya": "la altura en Goya",
    "altura_reconquista": "la altura en Reconquista",
    "var_reconquista_7d": "el cambio de Reconquista en los últimos 7 días",
}
CLASES = {0: "Normal", 1: "Alerta", 2: "Evacuación"}
SEMAFORO = {
    0: ("🟢", "#2e7d32", "#e8f5e9", "No se esperan niveles de alerta en los próximos 7 días."),
    1: ("🟡", "#b26a00", "#fff8e1", "El río podría superar el nivel de alerta (5,00 m) en los próximos 7 días."),
    2: ("🔴", "#c62828", "#ffebee", "El río podría superar el nivel de evacuación (5,25 m) en los próximos 7 días."),
}
ARCHIVO_MODELO = Path(__file__).parent / "modelo_simple_rosario.joblib"
PARAMETROS_ARBOL = dict(criterion="gini", max_depth=3, min_samples_leaf=10,
                        class_weight="balanced", random_state=42)


# --------------------------------------------------------------------------
# 2. Descarga de alturas del INA
# --------------------------------------------------------------------------
def pedir_serie(series_id: int, desde: str, hasta: str, intentos: int = 3) -> pd.Series:
    """Pide una serie al INA y devuelve la altura máxima de cada día."""
    params = {"tipo": "puntual", "series_id": series_id, "timestart": desde,
              "timeend": hasta, "format": "csvless", "no_id": "true"}
    texto = ""
    for i in range(intentos):
        try:
            r = requests.get(BASE_INA, params=params, timeout=60)
            if r.status_code == 200:
                texto = r.text
                break
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    if not texto.strip():
        return pd.Series(dtype=float)

    df = pd.read_csv(StringIO(texto), header=None)
    fechas = pd.to_datetime(df.iloc[:, 0], errors="coerce", utc=True)
    valores = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    obs = pd.DataFrame({"fecha": fechas, "valor": valores}).dropna()
    obs = obs[(obs["valor"] > -5) & (obs["valor"] < 40)]          # descarta errores de sensor
    obs["dia"] = (obs["fecha"].dt.tz_convert("America/Argentina/Buenos_Aires")
                  .dt.tz_localize(None).dt.normalize())
    return obs.groupby("dia")["valor"].max()


@st.cache_data(ttl=3600, show_spinner="Descargando alturas del INA...")
def alturas_recientes(dias: int = DIAS_GRAFICO) -> pd.DataFrame:
    """Alturas máximas diarias de las 4 estaciones en los últimos días."""
    hasta = date.today() + timedelta(days=1)
    desde = hasta - timedelta(days=dias + 15)
    series = {est: pedir_serie(sid, desde.isoformat(), hasta.isoformat())
              for est, sid in SERIES_INA.items()}
    return pd.DataFrame(series).sort_index()


# --------------------------------------------------------------------------
# 3. Modelo: se carga el .joblib del notebook (o se entrena si no está)
# --------------------------------------------------------------------------
def entrenar_desde_ina() -> dict:
    """Plan B: entrena el mismo árbol del notebook con todos los datos 1980–hoy."""
    hoy = pd.Timestamp(date.today())
    alturas = {}
    for est, sid in SERIES_INA.items():
        partes, inicio = [], pd.Timestamp("1980-01-01")
        while inicio < hoy:
            fin = min(inicio + pd.DateOffset(years=5), hoy)
            partes.append(pedir_serie(sid, inicio.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")))
            inicio = fin
        alturas[est] = pd.concat(partes).groupby(level=0).max()

    dias = pd.date_range("1980-01-01", hoy, freq="D")
    tabla = pd.DataFrame(index=dias)
    tabla["altura_hoy"] = alturas["Rosario"].reindex(dias)
    tabla["altura_corrientes"] = alturas["Corrientes"].reindex(dias)
    tabla["altura_goya"] = alturas["Goya"].reindex(dias)
    tabla["altura_reconquista"] = alturas["Reconquista"].reindex(dias)
    tabla["var_reconquista_7d"] = tabla["altura_reconquista"] - tabla["altura_reconquista"].shift(7)
    maximo_futuro = tabla["altura_hoy"].shift(-HORIZONTE).rolling(HORIZONTE, min_periods=4).max()
    tabla["clase"] = np.select([maximo_futuro >= EVACUACION, maximo_futuro >= ALERTA], [2, 1], 0)
    tabla.loc[maximo_futuro.isna(), "clase"] = np.nan
    datos = tabla.dropna()

    modelo = DecisionTreeClassifier(**PARAMETROS_ARBOL)
    modelo.fit(datos[VARIABLES], datos["clase"].astype(int))
    return {"modelo": modelo, "variables": VARIABLES, "umbrales": (ALERTA, EVACUACION),
            "horizonte": HORIZONTE, "clases": CLASES}


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo() -> dict:
    if ARCHIVO_MODELO.exists():
        paquete = joblib.load(ARCHIVO_MODELO)
        paquete["origen"] = "Archivo modelo_simple_rosario.joblib (entrenado en Colab)"
    else:
        paquete = entrenar_desde_ina()
        paquete["origen"] = "Entrenado al iniciar la app con datos del INA (no se encontró el .joblib)"
    return paquete


# --------------------------------------------------------------------------
# 4. Variables del último día y explicación de la decisión
# --------------------------------------------------------------------------
def armar_variables(alturas: pd.DataFrame) -> pd.DataFrame:
    """Arma las 5 variables para cada día y devuelve la tabla completa."""
    tabla = pd.DataFrame(index=alturas.index)
    tabla["altura_hoy"] = alturas["Rosario"]
    tabla["altura_corrientes"] = alturas["Corrientes"]
    tabla["altura_goya"] = alturas["Goya"]
    tabla["altura_reconquista"] = alturas["Reconquista"]
    reconquista = alturas["Reconquista"].asfreq("D")
    tabla["var_reconquista_7d"] = (reconquista - reconquista.shift(7)).reindex(tabla.index)
    return tabla


def m(valor: float) -> str:
    """Formatea metros con coma decimal: 5.25 -> '5,25 m'."""
    return f"{valor:.2f} m".replace(".", ",")


def camino_de_reglas(arbol: DecisionTreeClassifier, fila: pd.DataFrame, variables: list) -> list:
    """Traduce a palabras las preguntas que el árbol hizo para este día."""
    t = arbol.tree_
    pasos = []
    for nodo in arbol.decision_path(fila).indices:
        if t.children_left[nodo] == -1:          # hoja: no hay pregunta
            continue
        var = variables[t.feature[nodo]]
        umbral = t.threshold[nodo]
        valor = float(fila[var].iloc[0])
        signo = "≤" if valor <= umbral else ">"
        pasos.append((NOMBRES[var], valor, signo, umbral))
    return pasos


# --------------------------------------------------------------------------
# 5. Interfaz
# --------------------------------------------------------------------------
st.title("🌊 Alerta temprana de crecidas del río Paraná en Rosario")
st.caption("Árbol de clasificación entrenado con alturas del río 1980–2026 · Datos: Instituto Nacional del Agua (INA)")

with st.sidebar:
    st.header("Acerca de la app")
    st.write("Estima, con **7 días de anticipación**, si el río Paraná en Rosario estará en nivel "
             "**Normal**, **Alerta** o **Evacuación**, según los umbrales oficiales del INA "
             "(alerta 5,00 m · evacuación 5,25 m).")
    st.write("Usa 5 datos: la altura en Rosario, Corrientes, Goya y Reconquista, y el cambio "
             "de Reconquista en la última semana.")
    st.write("Evaluado en las crecidas de 1992, 1998, 2010 y 2016: no dejó ninguna crecida sin "
             "aviso y avisó en promedio 11,75 días antes.")
    if st.button("🔄 Actualizar datos del INA"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("Proyecto de Modelizado de Sistemas de IA (PPII) – Unidad 3 – IFTS N° 18")
    st.caption("Leanderson Carvalho · Stefania Silva · Guillermo Seron · Richard Quintero · Gonzalo Yahuita")

# Datos y modelo
try:
    alturas = alturas_recientes()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudieron descargar los datos del INA: {e}")
    st.stop()

faltan = [est for est in SERIES_INA if est not in alturas or alturas[est].dropna().empty]
if faltan:
    st.error("El INA no devolvió datos recientes de: " + ", ".join(faltan) +
             ". Probá más tarde con el botón «Actualizar datos del INA».")
    st.stop()

paquete = cargar_modelo()
modelo, variables = paquete["modelo"], paquete["variables"]

tabla = armar_variables(alturas)
completos = tabla[variables].dropna()
if completos.empty:
    st.error("No hay ningún día reciente con las 4 estaciones medidas. Probá más tarde.")
    st.stop()

fila = completos.iloc[[-1]]
fecha_dato = fila.index[0]
clase = int(modelo.predict(fila)[0])
probas = dict(zip([int(k) for k in modelo.classes_], modelo.predict_proba(fila)[0]))

# Semáforo
emoji, color, fondo, mensaje = SEMAFORO[clase]
st.markdown(
    f"""
    <div style="background:{fondo};border-left:10px solid {color};padding:18px 24px;border-radius:8px;">
      <div style="font-size:15px;color:#555;">Pronóstico para los próximos {HORIZONTE} días
      (datos del {fecha_dato:%d/%m/%Y})</div>
      <div style="font-size:42px;font-weight:700;color:{color};">{emoji} {CLASES[clase]}</div>
      <div style="font-size:17px;">{mensaje}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if (pd.Timestamp(date.today()) - fecha_dato).days > 3:
    st.warning(f"El último dato completo es del {fecha_dato:%d/%m/%Y}: el INA puede estar demorado.")

st.write("")
c1, c2, c3 = st.columns(3)
for col, k in zip((c1, c2, c3), (0, 1, 2)):
    col.metric(f"Probabilidad {CLASES[k]}", f"{100 * probas.get(k, 0):.0f} %")

# Alturas de las estaciones
st.subheader("Altura del río en las estaciones")
cols = st.columns(4)
for col, est in zip(cols, SERIES_INA):
    serie = alturas[est].dropna()
    hoy = serie.loc[:fecha_dato].iloc[-1]
    antes = serie.loc[:fecha_dato - pd.Timedelta(days=7)]
    delta = (f"{hoy - antes.iloc[-1]:+.2f} m en 7 días".replace(".", ",")) if len(antes) else None
    col.metric(est, m(hoy), delta, delta_color="inverse")  # subir = rojo

# Explicación
st.subheader("¿Por qué el modelo dice esto?")
st.write("El árbol de clasificación hizo estas preguntas, en este orden:")
for i, (nombre, valor, signo, umbral) in enumerate(camino_de_reglas(modelo, fila, variables), 1):
    st.markdown(f"**{i}.** ¿{nombre[0].upper() + nombre[1:]} es menor o igual a **{m(umbral)}**? "
                f"Es **{m(valor)}** → {'sí' if signo == '≤' else 'no'}.")
st.markdown(f"➡️ Resultado: **{CLASES[clase]}**.")

# Gráfico de Rosario
st.subheader(f"Río Paraná en Rosario: últimos {DIAS_GRAFICO} días")
rosario = alturas["Rosario"].dropna().tail(DIAS_GRAFICO).rename("altura").reset_index()
rosario.columns = ["fecha", "altura"]
linea = alt.Chart(rosario).mark_line(color="#1f4e79").encode(
    x=alt.X("fecha:T", title="Fecha"),
    y=alt.Y("altura:Q", title="Altura (m)",
            scale=alt.Scale(domain=[min(rosario["altura"].min() - 0.3, 0), EVACUACION + 0.5])),
    tooltip=[alt.Tooltip("fecha:T", title="Fecha"), alt.Tooltip("altura:Q", title="Altura (m)", format=".2f")],
)
umbrales = pd.DataFrame({"y": [ALERTA, EVACUACION], "nivel": ["Alerta 5,00 m", "Evacuación 5,25 m"]})
reglas = alt.Chart(umbrales).mark_rule(strokeDash=[6, 4]).encode(
    y="y:Q",
    color=alt.Color("nivel:N", scale=alt.Scale(range=["#f2a900", "#c62828"]), title="Umbral"),
)
st.altair_chart(linea + reglas, width="stretch")

with st.expander("Ver las alturas de las estaciones aguas arriba"):
    otras = alturas[["Corrientes", "Goya", "Reconquista"]].dropna(how="all").tail(DIAS_GRAFICO)
    st.line_chart(otras, y_label="Altura (m)")

with st.expander("Detalles del modelo"):
    st.write(f"**Origen del modelo:** {paquete.get('origen', '')}")
    st.write(f"**Profundidad del árbol:** {modelo.get_depth()} · **Hojas:** {modelo.get_n_leaves()}")
    st.write("**Variables:** " + ", ".join(variables))

st.divider()
st.caption("⚠️ Prototipo académico. No reemplaza las alertas oficiales del Instituto Nacional del Agua (INA), "
           "el Servicio Meteorológico Nacional ni Defensa Civil.")
