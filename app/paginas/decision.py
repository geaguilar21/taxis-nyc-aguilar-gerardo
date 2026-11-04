import streamlit as st

st.title("Decisión: cuántos taxis enviar")
st.write("Si quedarse corto cuesta más que enviar un taxi ocioso, conviene usar un cuantil superior al p50.")
cp = st.number_input("Costo de un pasajero perdido", min_value=0.0, value=40.0, step=5.0)
co = st.number_input("Costo de un taxi ocioso por hora", min_value=0.0, value=10.0, step=5.0)
q = cp / (cp + co) if cp + co else 0.5
st.metric("Cuantil óptimo", f"p{q:.0%}")
st.info("La API entrega cuantiles p10–p90; la recomendación usa el cuantil que corresponde a los costos indicados.")
st.latex(r"q^* = \frac{costo\ de\ pasajero\ perdido}{costo\ de\ pasajero\ perdido + costo\ de\ taxi\ ocioso}")
