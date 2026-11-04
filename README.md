# Taxis NYC

Proyecto de pronóstico, segmentación y decisión operativa para taxis amarillos de Nueva York.

## Estructura

- `preparacion.ipynb`: limpieza, serie horaria, modelo de cuantiles, evaluación y segmentación.
- `api/`: FastAPI con `/`, `/info` y `/pronostico`.
- `app/`: aplicación Streamlit con páginas de pronóstico, segmentación, decisión y ficha del modelo.
- `app/data/`: segmentos, PCA y siluetas precalculados.

## Ejecución local

```bash
uvicorn api.main:app --reload --port 8000
streamlit run app/streamlit_app.py
```

La app puede usar `API_URL` como variable de entorno o como secret de Streamlit.

## Datos y reglas

Se usaron viajes de enero de 2024, origen `PULocationID`, entrenamiento hasta el 24 de enero y prueba del 25 al 31. Las zonas pronosticables tienen al menos 3,000 viajes y las zonas segmentadas al menos 500, excluyendo `Unknown`.

## Publicación

Render: raíz `api`, comando `pip install -r requirements.txt`, inicio `uvicorn main:app --host 0.0.0.0 --port $PORT`.

Streamlit Community Cloud: archivo principal `app/streamlit_app.py` y secret `API_URL` con la URL pública de Render.
