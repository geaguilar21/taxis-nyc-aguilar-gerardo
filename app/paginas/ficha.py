import json
from pathlib import Path
import streamlit as st

st.title("Ficha del modelo")
metrics = json.loads(Path(__file__).resolve().parents[2].joinpath("metrics.json").read_text(encoding="utf-8"))
st.metric("MAE del p50", f"{metrics['metricas']['mae_p50']:.2f} viajes")
st.metric("MAE igual que hace una semana", f"{metrics['metricas']['mae_hace_una_semana']:.2f} viajes")
st.metric("Cobertura p10–p90", f"{metrics['metricas']['cobertura_p10_p90']:.1%}")
st.markdown("""
### Uso recomendado
Apoyar la planificación diaria de flota en las zonas que superan el mínimo de viajes del proyecto. La app debe mostrar la banda de incertidumbre y no solo una cifra puntual.

### Limitaciones
El histórico cubre únicamente enero de 2024. Las zonas con pocos viajes no tienen pronóstico, los cuantiles no garantizan cobertura perfecta en otros meses y la regla de una carrera por taxi es una simplificación operativa.

### Comunicación ética
Una mala predicción puede dejar sin servicio a un barrio o generar taxis ociosos. El modelo no debe usarse para retirar permanentemente el servicio de zonas con baja demanda observada.
""")
