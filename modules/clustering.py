"""Clustering automatico: K-means y DBSCAN sobre variables numericas.

seleccion automatica de un valor razonable de k para K-means
(metodo del codo y silhouette) y clustering por densidad con DBSCAN, con
sugerencia automatica de eps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_K_RANGE = range(2, 11)
DEFAULT_MIN_SAMPLES = 5
DEFAULT_DBSCAN_EPS = 0.5
# El punto de codo detectado tiende a subestimar el radio real cuando los
# datos no tienen ruido (no hay una rodilla marcada porque no hay puntos que
# "sobresalgan"); un margen del 50% evita marcar como ruido a puntos que en
# realidad son el borde normal de un cluster denso.
EPS_SAFETY_MARGIN = 1.5
# silhouette_score es O(n^2) en memoria/tiempo; se acota la muestra para que
# datasets grandes (decenas/cientos de miles de filas, ver RNF-01) no se cuelguen.
SILHOUETTE_SAMPLE_SIZE = 5000
# A diferencia de K-means (lineal en n), el costo de DBSCAN crece mucho mas
# rapido que lineal con la cantidad de filas cuando los clusters son densos:
# medido empiricamente con eps altos, 100k filas ya rondaba ~19-22s (poco
# margen frente al limite de 30s del RNF-01), mientras que 75k se mantiene
# comodamente entre 8-12s. Por eso se limita a una muestra.
DBSCAN_MAX_ROWS = 75_000


@dataclass
class KSelection:
    """Resultado de explorar varios valores de k para K-means."""

    k_values: list[int]
    inertias: list[float]
    silhouette_scores: list[float]
    best_k_elbow: int
    best_k_silhouette: int


@dataclass
class ClusteringResult:
    """Etiquetas de cluster alineadas al DataFrame original, y metadatos del modelo."""

    method: str
    labels: pd.Series
    n_clusters: int
    params: dict


def _numeric_subset(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    return df[numeric_columns].dropna()


def _scale(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    subset = _numeric_subset(df, numeric_columns)
    scaled = StandardScaler().fit_transform(subset)
    return scaled, subset


def _scale_for_dbscan(
    df: pd.DataFrame, numeric_columns: list[str], max_rows: int | None = None
) -> tuple[np.ndarray, pd.DataFrame, bool]:
    """Como `_scale`, pero toma una muestra reproducible si hay mas de `max_rows` filas."""
    if max_rows is None:
        max_rows = DBSCAN_MAX_ROWS  # se resuelve en llamada, no en definicion, para poder testear con monkeypatch

    subset = _numeric_subset(df, numeric_columns)
    was_sampled = len(subset) > max_rows
    sampled = subset.sample(n=max_rows, random_state=RANDOM_STATE) if was_sampled else subset
    scaled = StandardScaler().fit_transform(sampled)
    return scaled, sampled, was_sampled


def _find_elbow(x_values: list[float], y_values: list[float]) -> float:
    """Punto de maxima curvatura: el mas alejado de la recta que une los extremos."""
    if len(x_values) < 3:
        return x_values[0]

    points = np.array(list(zip(x_values, y_values)), dtype=float)
    ranges = points.max(axis=0) - points.min(axis=0)
    ranges[ranges == 0] = 1.0  # evita division por cero si un eje es constante
    normalized = (points - points.min(axis=0)) / ranges

    start, end = normalized[0], normalized[-1]
    line_vec = end - start
    line_vec /= np.linalg.norm(line_vec)

    distances = []
    for point in normalized:
        vec = point - start
        projection = np.dot(vec, line_vec) * line_vec
        distances.append(np.linalg.norm(vec - projection))

    return x_values[int(np.argmax(distances))]


def select_k(df: pd.DataFrame, numeric_columns: list[str], k_range: range = DEFAULT_K_RANGE) -> KSelection:
    """Explora varios valores de k y sugiere uno por metodo del codo y por silhouette."""
    scaled, subset = _scale(df, numeric_columns)
    k_values = [k for k in k_range if k < len(subset)]

    if not k_values:
        return KSelection([], [], [], best_k_elbow=2, best_k_silhouette=2)

    inertias: list[float] = []
    silhouettes: list[float] = []
    sample_size = min(len(scaled), SILHOUETTE_SAMPLE_SIZE)

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = model.fit_predict(scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(
            float(silhouette_score(scaled, labels, sample_size=sample_size, random_state=RANDOM_STATE))
        )

    best_k_elbow = int(_find_elbow(k_values, inertias))
    best_k_silhouette = k_values[int(np.argmax(silhouettes))]

    return KSelection(k_values, inertias, silhouettes, best_k_elbow, best_k_silhouette)


def run_kmeans(df: pd.DataFrame, numeric_columns: list[str], k: int) -> ClusteringResult:
    """Ejecuta K-means con k clusters sobre columnas numericas estandarizadas."""
    scaled, subset = _scale(df, numeric_columns)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    predicted = model.fit_predict(scaled)

    labels = pd.Series(-1, index=df.index, name="cluster")
    labels.loc[subset.index] = predicted
    return ClusteringResult("kmeans", labels, n_clusters=k, params={"k": k})


def suggest_dbscan_eps(df: pd.DataFrame, numeric_columns: list[str], min_samples: int = DEFAULT_MIN_SAMPLES) -> float:
    """Sugiere eps a partir del 'codo' en las distancias al min_samples-esimo vecino mas cercano."""
    scaled, _, _ = _scale_for_dbscan(df, numeric_columns)
    if len(scaled) <= min_samples:
        return DEFAULT_DBSCAN_EPS

    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(scaled)
    distances, _ = neighbors.kneighbors(scaled)
    k_distances = np.sort(distances[:, -1])

    positions = list(range(len(k_distances)))
    elbow_position = int(_find_elbow(positions, k_distances.tolist()))
    return float(k_distances[elbow_position]) * EPS_SAFETY_MARGIN


def run_dbscan(
    df: pd.DataFrame, numeric_columns: list[str], eps: float, min_samples: int = DEFAULT_MIN_SAMPLES
) -> ClusteringResult:
    """Ejecuta DBSCAN sobre columnas numericas estandarizadas. Etiqueta -1 = ruido.

    En datasets muy grandes (> DBSCAN_MAX_ROWS) se ajusta sobre una muestra
    aleatoria reproducible por rendimiento (RNF-01); las filas fuera de la
    muestra quedan con etiqueta -1 y `params['sampled']` queda en True para
    que la UI lo pueda avisar.
    """
    scaled, subset, was_sampled = _scale_for_dbscan(df, numeric_columns)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    predicted = model.fit_predict(scaled)

    labels = pd.Series(-1, index=df.index, name="cluster")
    labels.loc[subset.index] = predicted
    n_clusters = len(set(predicted) - {-1})
    params = {
        "eps": eps,
        "min_samples": min_samples,
        "sampled": was_sampled,
        "sample_size": len(subset) if was_sampled else None,
    }
    return ClusteringResult("dbscan", labels, n_clusters=n_clusters, params=params)


def cluster_summary(df: pd.DataFrame, labels: pd.Series, numeric_columns: list[str]) -> pd.DataFrame:
    """Cantidad de filas y promedio de cada variable numerica por cluster (-1 = ruido en DBSCAN)."""
    data = df[numeric_columns].copy()
    data["cluster"] = labels
    grouped = data.groupby("cluster")
    summary = grouped[numeric_columns].mean()
    summary.insert(0, "n_filas", grouped.size())
    return summary
