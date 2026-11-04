import os
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000")).rstrip("/")

@st.cache_data(ttl=3600)
def get_info():
    return requests.get(f"{API_URL}/info", timeout=90).json()

@st.cache_data(ttl=3600)
def get_forecast(zone, day):
    return requests.get(f"{API_URL}/pronostico", params={"zona_id": zone, "fecha": day}, timeout=90).json()

st.title("Pronóstico de demanda")
st.caption("La mediana es p50; la banda representa los cuantiles p10–p90.")
try:
    info = get_info()
except requests.RequestException:
    st.error("La API no responde todavía. Revisa el enlace o espera a que Render despierte.")
    st.stop()
zonas = pd.DataFrame(info["zonas"])
if zonas.empty:
    st.warning("La API no devolvió zonas pronosticables.")
    st.stop()
zona_id = st.selectbox("Zona", zonas["PULocationID"], format_func=lambda x: zonas.loc[zonas.PULocationID.eq(x), "nombre"].iloc[0])
fecha = st.selectbox("Fecha", info["fechas_disponibles"])
try:
    data = get_forecast(int(zona_id), fecha)
except requests.RequestException:
    st.error("No se pudo consultar el pronóstico.")
    st.stop()
df = pd.DataFrame(data["horas"])
df["p10"] = df["cuantiles"].apply(lambda x: x.get("p10"))
df["p50"] = df["cuantiles"].apply(lambda x: x.get("p50", x.get("p10")))
df["p90"] = df["cuantiles"].apply(lambda x: x.get("p90", x.get("p10")))
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.hora, y=df.p90, line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=df.hora, y=df.p10, fill="tonexty", fillcolor="rgba(124,58,237,.18)", line=dict(width=0), name="p10–p90"))
fig.add_trace(go.Scatter(x=df.hora, y=df.p50, mode="lines+markers", name="p50", line=dict(color="#7c3aed", width=3)))
if df["real"].notna().any():
    fig.add_trace(go.Scatter(x=df.hora, y=df.real, mode="markers", name="real", marker=dict(color="#111827")))
fig.update_layout(xaxis_title="Hora", yaxis_title="Viajes", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df[["hora", "p10", "p50", "p90", "real", "hace_una_semana"]], use_container_width=True, hide_index=True)
