import io
import os
import zipfile
import tempfile
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
st.set_page_config(page_title="Catastro Bogotá | Regresión", page_icon="🏙️", layout="wide")

st.title("🏙️ Explorador de modelos de avalúo catastral")
st.caption("Proyecto académico · Catastro Bogotá · Avalúo mediano por manzana · Vigencia 2023")
st.info(
    "Esta aplicación estima el valor mediano del avalúo catastral de una **manzana**. "
    "No calcula el avalúo oficial de un predio individual ni reemplaza a la autoridad catastral."
)

@st.cache_data(show_spinner=False)
def load_zip(raw_bytes):
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "dataset.zip")
        with open(zpath, "wb") as f:
            f.write(raw_bytes)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(tmp)
        shp_files = []
        for root, _, files in os.walk(tmp):
            shp_files += [os.path.join(root, f) for f in files if f.lower().endswith(".shp")]
        if not shp_files:
            raise ValueError("El ZIP no contiene un archivo .shp.")
        gdf = gpd.read_file(shp_files[0])
    required = {"MANCODIGO", "MED_VALOR_", "NO_PREDIOS"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Faltan campos requeridos: {sorted(missing)}")
    gdf["MANCODIGO"] = gdf["MANCODIGO"].astype(str).str.strip()
    attrs = [c for c in gdf.columns if c not in ["MANCODIGO", "geometry"]]
    agg = {c: "first" for c in attrs}
    gdf = gdf.dissolve(by="MANCODIGO", as_index=False, aggfunc=agg)
    if gdf.crs is None:
        raise ValueError("El shapefile no declara sistema de coordenadas (CRS).")
    metric = gdf.to_crs(3116)
    centroid = metric.geometry.centroid.to_crs(4326)
    df = pd.DataFrame({
        "MANCODIGO": gdf["MANCODIGO"].astype(str),
        "valor_catastral": pd.to_numeric(gdf["MED_VALOR_"], errors="coerce"),
        "num_predios": pd.to_numeric(gdf["NO_PREDIOS"], errors="coerce"),
        "area_manzana": metric.geometry.area,
        "longitud": centroid.x,
        "latitud": centroid.y,
    })
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["valor_catastral"])
    df = df[df["valor_catastral"] > 0].reset_index(drop=True)
    return df

def make_pipeline(model):
    numeric = ["num_predios", "area_manzana", "latitud", "longitud"]
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric)
    ])
    return Pipeline([("prep", prep), ("model", model)])

def metrics(y, pred):
    return {
        "MAE (COP)": mean_absolute_error(y, pred),
        "RMSE (COP)": np.sqrt(mean_squared_error(y, pred)),
        "R²": r2_score(y, pred)
    }

@st.cache_resource(show_spinner="Entrenando y evaluando los modelos...")
def train_all(data, max_rows):
    df = data.copy()
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=SEED).reset_index(drop=True)
    features = ["num_predios", "area_manzana", "latitud", "longitud"]
    X = df[features]
    y = df["valor_catastral"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    specs = {
        "Regresión lineal": (
            LinearRegression(), None
        ),
        "Árbol de decisión": (
            DecisionTreeRegressor(random_state=SEED),
            {"model__max_depth": [4, 8, 12, None],
             "model__min_samples_leaf": [5, 15, 30]}
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [8, 15, None],
             "model__min_samples_leaf": [3, 8, 15],
             "model__max_features": [0.7, 1.0]}
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=SEED),
            {"model__n_estimators": [100, 200],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [2, 3],
             "model__min_samples_leaf": [5, 15]}
        )
    }
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error"
    }
    models, rows, cv_rows = {}, [], []
    baseline = make_pipeline(DummyRegressor(strategy="mean")).fit(X_train, y_train)
    btr = metrics(y_train, baseline.predict(X_train))
    bte = metrics(y_test, baseline.predict(X_test))
    rows.append({"Modelo":"Referencia: promedio", **{f"Entrenamiento · {k}":v for k,v in btr.items()},
                 **{f"Prueba · {k}":v for k,v in bte.items()}, "Brecha R² (train-test)":btr["R²"]-bte["R²"]})
    models["Referencia: promedio"] = baseline

    for name, (estimator, params) in specs.items():
        pipe = make_pipeline(estimator)
        if params:
            search = RandomizedSearchCV(
                pipe, params, n_iter=8, scoring="neg_root_mean_squared_error",
                cv=cv, random_state=SEED, n_jobs=-1, refit=True,
                return_train_score=True
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            best = search.best_params_
        else:
            model = pipe.fit(X_train, y_train)
            best = {}
        models[name] = model
        tr = metrics(y_train, model.predict(X_train))
        te = metrics(y_test, model.predict(X_test))
        rows.append({"Modelo":name, **{f"Entrenamiento · {k}":v for k,v in tr.items()},
                     **{f"Prueba · {k}":v for k,v in te.items()}, "Brecha R² (train-test)":tr["R²"]-te["R²"]})
        cv_result = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                                   return_train_score=False, n_jobs=-1)
        cv_rows.append({
            "Modelo": name,
            "R² CV media": np.mean(cv_result["test_r2"]),
            "R² CV desv.": np.std(cv_result["test_r2"]),
            "MAE CV (COP)": -np.mean(cv_result["test_mae"]),
            "RMSE CV (COP)": -np.mean(cv_result["test_rmse"]),
            "Mejores parámetros": str(best) if best else "Sin ajuste"
        })
    return df, X_train, X_test, y_train, y_test, models, pd.DataFrame(rows).set_index("Modelo"), pd.DataFrame(cv_rows)

with st.sidebar:
    st.header("Datos y configuración")
    uploaded = st.file_uploader("Sube avalcatmz2023.zip", type=["zip"])
    max_rows = st.select_slider("Filas para entrenar", options=[5000, 8000, 10000, 15000, 25000, 37430], value=10000)
    st.caption("La muestra se selecciona con semilla fija (42). Todos los modelos reciben la misma muestra y división 80/20.")

if uploaded is None:
    st.warning("Para iniciar, carga el archivo `avalcatmz2023.zip` en el panel lateral.")
    st.markdown("""
    **El ZIP debe contener el shapefile completo**, con sus archivos `.shp`, `.shx`, `.dbf` y `.prj`.

    Campos esperados: `MANCODIGO`, `MED_VALOR_` y `NO_PREDIOS`.
    """)
    st.stop()

try:
    data = load_zip(uploaded.getvalue())
    st.success(f"Base cargada: {len(data):,} manzanas después de consolidar códigos y validar el avalúo.")
except Exception as e:
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

with st.spinner("Preparando los datos y calculando métricas comparables..."):
    df, X_train, X_test, y_train, y_test, models, comparison, cv_table = train_all(data, max_rows)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Manzanas válidas", f"{len(data):,}")
m2.metric("Entrenamiento", f"{len(X_train):,} · 80%")
m3.metric("Prueba", f"{len(X_test):,} · 20%")
m4.metric("Vigencia", "2023")

st.subheader("1. Comparación justa: entrenamiento frente a prueba")
st.write("Los cuatro modelos usan las mismas variables, la misma muestra y exactamente los mismos índices de entrenamiento y prueba.")
display = comparison.copy()
for c in display.columns:
    if "COP" in c:
        display[c] = display[c].map(lambda v: f"${v:,.0f}")
    elif "R²" in c or "Brecha" in c:
        display[c] = display[c].map(lambda v: f"{v:.4f}")
st.dataframe(display, use_container_width=True)

st.caption("MAE y RMSE se expresan en pesos colombianos. R² no es una exactitud porcentual; puede ser negativo. Una brecha train-test grande puede indicar sobreajuste.")

st.subheader("2. Validación cruzada sobre el 80% de entrenamiento")
cv_show = cv_table.copy()
for c in ["R² CV media", "R² CV desv."]:
    cv_show[c] = cv_show[c].map(lambda v: f"{v:.4f}")
for c in ["MAE CV (COP)", "RMSE CV (COP)"]:
    cv_show[c] = cv_show[c].map(lambda v: f"${v:,.0f}")
st.dataframe(cv_show, use_container_width=True)
st.caption("La validación cruzada de 5 particiones se ejecuta solo dentro del 80% de entrenamiento. El 20% de prueba queda reservado para la evaluación final.")

st.subheader("3. Visualización de R²")
fig, ax = plt.subplots(figsize=(9, 4))
names = comparison.index.tolist()
x = np.arange(len(names))
width = 0.36
ax.bar(x-width/2, comparison["Entrenamiento · R²"], width, label="Entrenamiento")
ax.bar(x+width/2, comparison["Prueba · R²"], width, label="Prueba")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("R²")
ax.set_title("R² en los mismos modelos y partición")
ax.legend()
st.pyplot(fig, clear_figure=True)

st.subheader("4. Explorar una manzana catastral")
st.write("Ajusta las características de una manzana para obtener una estimación experimental. Las coordenadas deben estar dentro del área de estudio.")
c1, c2, c3 = st.columns(3)
with c1:
    num_predios = st.number_input("Número de predios", min_value=1, max_value=10000, value=int(data.num_predios.median()))
    area_manzana = st.number_input("Área de manzana (m²)", min_value=1.0, value=float(data.area_manzana.median()), step=10.0)
with c2:
    latitud = st.number_input("Latitud", min_value=3.8, max_value=4.9, value=float(data.latitud.median()), format="%.6f")
    longitud = st.number_input("Longitud", min_value=-74.5, max_value=-73.8, value=float(data.longitud.median()), format="%.6f")
with c3:
    model_name = st.selectbox("Modelo para estimar", list(models.keys()))
    st.write("")
    st.write("")
    predict = st.button("Calcular estimación", type="primary", use_container_width=True)

if predict:
    sample = pd.DataFrame([{
        "num_predios":num_predios, "area_manzana":area_manzana,
        "latitud":latitud, "longitud":longitud
    }])
    estimate = models[model_name].predict(sample)[0]
    st.metric("Estimación experimental del avalúo mediano", f"${estimate:,.0f} COP")
    st.caption("Estimación del modelo para una manzana con esos atributos; no es un avalúo oficial.")

st.subheader("5. Revisar errores del mejor modelo")
eligible = comparison.drop(index=["Referencia: promedio"], errors="ignore")
best_name = eligible["Prueba · RMSE (COP)"].idxmin()
best_model = models[best_name]
pred = best_model.predict(X_test)
errors = pd.DataFrame({"Real":y_test, "Predicho":pred})
errors["Error absoluto (COP)"] = np.abs(errors["Real"]-errors["Predicho"])
errors["Error relativo (%)"] = errors["Error absoluto (COP)"]/errors["Real"]*100
st.write(f"Menor RMSE en prueba: **{best_name}**")
st.dataframe(errors.describe().T, use_container_width=True)

with st.expander("Diccionario de variables y limitaciones"):
    st.markdown("""
    - **MED_VALOR_**: variable objetivo usada como valor mediano del avalúo de la manzana (COP).
    - **NO_PREDIOS**: número de predios asociado a la manzana.
    - **Área de manzana**: área geométrica calculada en m² tras proyectar a EPSG:3116.
    - **Latitud y longitud**: centroide geométrico transformado a EPSG:4326.

    **Limitaciones:** la base no contiene área construida, antigüedad, estrato ni uso del suelo en este archivo. 
    Por ello el modelo usa solo cuatro predictores, dos de ellos coordenadas. El desempeño puede reflejar 
    patrones espaciales propios de Bogotá y no debe extrapolarse a otros municipios o años. Una división 
    aleatoria puede ser optimista por la cercanía espacial entre manzanas; conviene añadir una evaluación 
    espacial por zonas como análisis complementario.
    """)
