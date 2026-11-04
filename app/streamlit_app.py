import os
import streamlit as st

st.set_page_config(page_title="Taxis NYC", page_icon="🚕", layout="wide")
pages = [
    st.Page("paginas/pronostico.py", title="Pronóstico", icon="📈"),
    st.Page("paginas/segmentacion.py", title="Segmentación", icon="🗺️"),
    st.Page("paginas/decision.py", title="Decisión", icon="🚕"),
    st.Page("paginas/ficha.py", title="Ficha del modelo", icon="📋"),
]
pg = st.navigation(pages)
pg.run()
