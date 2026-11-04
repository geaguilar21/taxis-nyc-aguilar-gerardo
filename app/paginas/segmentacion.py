import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

st.title("Segmentación de zonas")
DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(DATA / "zonas_segmentadas.csv")
silueta = pd.read_csv(DATA / "silueta.csv")
st.caption("Las zonas se agruparon con variables de volumen, horario, distancia, tarifa y propina, después de escalar los datos.")
st.subheader("Elección de k")
st.line_chart(silueta.set_index("k")["silueta"])
grupos = st.multiselect("Grupos", sorted(df["grupo_nombre"].dropna().unique()), default=sorted(df["grupo_nombre"].dropna().unique()))
view = df[df["grupo_nombre"].isin(grupos)]
fig = px.scatter(view, x="pca_1", y="pca_2", color="grupo_nombre", hover_name="nombre", hover_data=["distrito", "viajes"])
st.plotly_chart(fig, use_container_width=True)
st.dataframe(view[["PULocationID", "nombre", "distrito", "grupo_nombre", "viajes", "distancia_mediana", "tarifa_mediana"]], use_container_width=True, hide_index=True)
