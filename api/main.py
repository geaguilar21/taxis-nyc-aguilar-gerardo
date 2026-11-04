from datetime import date
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DEMANDA = pd.read_parquet(BASE / "demanda.parquet")
DEMANDA["timestamp"] = pd.to_datetime(DEMANDA["timestamp"])
ZONAS_PATH = BASE / "data" / "zonas_segmentadas.csv"
if not ZONAS_PATH.exists():
    ZONAS_PATH = ROOT / "app" / "data" / "zonas_segmentadas.csv"
ZONAS = pd.read_csv(ZONAS_PATH)
INFO = json.loads((BASE / "info.json").read_text(encoding="utf-8"))
FEATURES = ["PULocationID", "lag_24", "lag_48", "lag_167", "lag_168", "lag_169", "media_dia_ant", "hora", "dia_semana", "feriado"]
CUANTILES = [int(q * 100) for q in INFO["cuantiles"]]

modelo = XGBRegressor()
modelo.load_model(str(BASE / "modelo_pronostico.ubj"))
app = FastAPI(title="Taxis NYC API", version="1.0.0")

@app.get("/")
def root():
    return {"estado": "en línea", "servicio": "pronóstico de demanda de taxis"}

@app.get("/info")
def info():
    zonas = ZONAS[ZONAS["PULocationID"].isin(INFO["zonas_pronosticables"])][["PULocationID", "nombre", "distrito"]].drop_duplicates()
    return {"zonas": zonas.to_dict(orient="records"), "fechas_disponibles": sorted(DEMANDA["timestamp"].dt.date.astype(str).unique().tolist()), "cuantiles": CUANTILES, "metricas": INFO["metricas"]}

@app.get("/pronostico")
def pronostico(zona_id: int, fecha: date):
    filas = DEMANDA[(DEMANDA["PULocationID"] == zona_id) & (DEMANDA["timestamp"].dt.date == fecha)]
    if filas.empty:
        raise HTTPException(status_code=404, detail="No hay datos para esa zona o fecha")
    pred = np.asarray(modelo.predict(filas[FEATURES]))
    if pred.ndim == 1:
        pred = pred[:, None]
    pred = np.sort(pred, axis=1)
    zona = ZONAS[ZONAS["PULocationID"] == zona_id].head(1)
    if zona.empty:
        raise HTTPException(status_code=404, detail="Zona no pronosticable")
    nombre = zona.iloc[0]
    horas = []
    for i, (_, fila) in enumerate(filas.sort_values("timestamp").iterrows()):
        cuantiles = {f"p{q}": round(float(pred[i, j]), 1) for j, q in enumerate(CUANTILES[:pred.shape[1]])}
        horas.append({"hora": int(fila["timestamp"].hour), "real": None if pd.isna(fila["real"]) else float(fila["real"]), "hace_una_semana": float(fila["hace_una_semana"]), "cuantiles": cuantiles})
    return {"zona": {"id": int(zona_id), "nombre": nombre["nombre"], "distrito": nombre["distrito"]}, "fecha": str(fecha), "horas": horas}
