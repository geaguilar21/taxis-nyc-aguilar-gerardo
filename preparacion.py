from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
RAW_TRIPS = DATA_ROOT / "yellow_tripdata_2024-01.parquet"
RAW_ZONES = DATA_ROOT / "taxi_zone_lookup.csv"
API_DIR = ROOT / "api"
APP_DATA = ROOT / "app" / "data"
API_DIR.mkdir(parents=True, exist_ok=True)
APP_DATA.mkdir(parents=True, exist_ok=True)

QUANTILES = np.arange(0.1, 1.0, 0.1)
FEATURES = ["PULocationID", "lag_24", "lag_48", "lag_167", "lag_168", "lag_169", "media_dia_ant", "hora", "dia_semana", "feriado"]


def load_trips():
    trips = pd.read_parquet(RAW_TRIPS, columns=["tpep_pickup_datetime", "PULocationID", "trip_distance", "fare_amount", "tip_amount"])
    zones = pd.read_csv(RAW_ZONES)
    valid_zones = set(zones.loc[zones["Borough"].ne("Unknown"), "LocationID"].astype(int))
    trips = trips.loc[
        trips["tpep_pickup_datetime"].between("2024-01-01", "2024-01-31 23:59:59")
        & trips["PULocationID"].isin(valid_zones)
    ].copy()
    trips["hora"] = trips["tpep_pickup_datetime"].dt.floor("h")
    return trips, zones


def make_series(trips):
    counts = trips.groupby(["PULocationID", "hora"]).size().rename("viajes")
    eligible = counts.groupby(level=0).sum()
    eligible = eligible[eligible >= 3000].index.astype(int).tolist()
    hours = pd.date_range("2024-01-01", "2024-02-01 23:00", freq="h")
    idx = pd.MultiIndex.from_product([eligible, hours], names=["PULocationID", "hora"])
    series = counts.reindex(idx, fill_value=0).rename("viajes").reset_index()
    g = series.groupby("PULocationID", group_keys=False)
    for lag in [24, 48, 167, 168, 169]:
        series[f"lag_{lag}"] = g["viajes"].shift(lag)
    series["media_dia_ant"] = g["viajes"].transform(lambda s: s.shift(24).rolling(24, min_periods=24).mean())
    series["hora_num"] = series["hora"].dt.hour
    series["hora"] = series["hora"].dt.tz_localize(None)
    series["hora"] = series["hora"].astype("datetime64[ns]")
    series["hora_del_dia"] = series["hora"].dt.hour
    series["dia_semana"] = series["hora"].dt.dayofweek
    series["feriado"] = series["hora"].dt.date.eq(pd.Timestamp("2024-01-15").date()).astype(int)
    return series, eligible


def train_forecast(series):
    work = series.dropna(subset=["lag_24", "lag_48", "lag_167", "lag_168", "lag_169", "media_dia_ant"]).copy()
    work["timestamp"] = series.loc[work.index, "hora"].values
    work["hora"] = work["hora_del_dia"]
    train = work.loc[work["timestamp"].between("2024-01-01", "2024-01-24 23:00")]
    test = work.loc[work["timestamp"].between("2024-01-25", "2024-01-31 23:00")]
    future = work.loc[work["timestamp"].between("2024-02-01", "2024-02-01 23:00")]
    model = XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=QUANTILES.tolist(),
        n_estimators=250, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, tree_method="hist",
        n_jobs=4, random_state=42,
    )
    model.fit(train[FEATURES], train["viajes"])
    pred_test = np.asarray(model.predict(test[FEATURES]))
    pred_future = np.asarray(model.predict(future[FEATURES]))
    if pred_test.ndim == 1:
        pred_test = pred_test[:, None]
        pred_future = pred_future[:, None]
    pred_test = np.sort(pred_test, axis=1)
    pred_future = np.sort(pred_future, axis=1)
    p50 = pred_test[:, 4 if pred_test.shape[1] >= 9 else 0]
    real = test["viajes"].to_numpy()
    week = test["lag_168"].to_numpy()
    metrics = {
        "mae_p50": float(mean_absolute_error(real, p50)),
        "mae_hace_una_semana": float(mean_absolute_error(real, week)),
        "cobertura_p10_p90": float(np.mean((real >= pred_test[:, 0]) & (real <= pred_test[:, -1]))),
        "filas_entrenamiento": int(len(train)),
        "filas_prueba": int(len(test)),
    }
    demand = pd.concat([test, future], ignore_index=True)
    preds = np.vstack([pred_test, pred_future])
    for i, q in enumerate(QUANTILES[:preds.shape[1]]):
        demand[f"p{int(q*100)}"] = preds[:, i]
    demand["real"] = demand["viajes"].where(demand["timestamp"] < pd.Timestamp("2024-02-01"), np.nan)
    demand["hace_una_semana"] = demand["lag_168"]
    demand.to_parquet(API_DIR / "demanda.parquet", index=False)
    model.save_model(str(API_DIR / "modelo_pronostico.ubj"))
    return model, demand, metrics


def segment_zones(trips, zones):
    t = trips.copy()
    t["hora_num"] = t["tpep_pickup_datetime"].dt.hour
    t["fin_de_semana"] = t["tpep_pickup_datetime"].dt.dayofweek >= 5
    features = t.groupby("PULocationID").agg(
        viajes=("PULocationID", "size"), distancia_mediana=("trip_distance", lambda s: s[s > 0].median()),
        tarifa_mediana=("fare_amount", lambda s: s[s > 0].median()), propina_mediana=("tip_amount", "median"),
        pct_fin_de_semana=("fin_de_semana", "mean"),
    ).reset_index()
    features["pct_6_10"] = t["hora_num"].between(6, 10).groupby(t["PULocationID"]).mean().values
    features["pct_16_20"] = t["hora_num"].between(16, 20).groupby(t["PULocationID"]).mean().values
    features["pct_22_5"] = ((t["hora_num"] >= 22) | (t["hora_num"] <= 5)).groupby(t["PULocationID"]).mean().values
    features = features[features["viajes"] >= 500].copy()
    features["log_viajes_dia"] = np.log1p(features["viajes"] / 31)
    metric_cols = ["log_viajes_dia", "distancia_mediana", "tarifa_mediana", "propina_mediana", "pct_fin_de_semana", "pct_6_10", "pct_16_20", "pct_22_5"]
    features[metric_cols] = features[metric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[metric_cols])
    silhouette = []
    for k in range(2, 8):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(scaled)
        silhouette.append({"k": k, "silueta": float(silhouette_score(scaled, labels))})
    best_k = max(silhouette, key=lambda x: x["silueta"])["k"]
    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    features["grupo"] = km.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(scaled)
    features["pca_1"], features["pca_2"] = xy[:, 0], xy[:, 1]
    profiles = features.groupby("grupo")[metric_cols].mean()
    names = {}
    for g, row in profiles.iterrows():
        if row["log_viajes_dia"] == profiles["log_viajes_dia"].max(): names[g] = "Alta demanda"
        elif row["pct_22_5"] == profiles["pct_22_5"].max(): names[g] = "Vida nocturna"
        elif row["distancia_mediana"] == profiles["distancia_mediana"].max(): names[g] = "Viajes largos"
        else: names[g] = "Demanda mixta"
    features["grupo_nombre"] = features["grupo"].map(names)
    features = features.merge(zones[["LocationID", "Zone", "Borough"]], left_on="PULocationID", right_on="LocationID", how="left")
    features.rename(columns={"Zone": "nombre", "Borough": "distrito"}, inplace=True)
    features.to_csv(APP_DATA / "zonas_segmentadas.csv", index=False)
    pd.DataFrame(silhouette).to_csv(APP_DATA / "silueta.csv", index=False)
    return features, silhouette, pca.explained_variance_ratio_.tolist()


def build_all():
    trips, zones = load_trips()
    series, eligible = make_series(trips)
    model, demand, metrics = train_forecast(series)
    segmented, silhouette, explained = segment_zones(trips, zones)
    info = {"zonas_pronosticables": eligible, "cuantiles": QUANTILES.tolist(), "metricas": metrics, "pca_varianza": explained}
    (API_DIR / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (ROOT / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
