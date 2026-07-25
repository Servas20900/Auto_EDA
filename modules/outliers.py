"""Deteccion de valores atipicos con Z-score, IQR e Isolation Forest.

Cubre RF-07: los tres metodos operan sobre las columnas numericas y sus
resultados se pueden comparar entre si (misma fila del DataFrame original,
marcada como atipica o no por cada metodo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

Z_SCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5
ISOLATION_FOREST_CONTAMINATION = 0.05
RANDOM_STATE = 42


@dataclass
class OutlierResult:
    """Mascara booleana de outliers para un metodo, alineada al indice original."""

    method: str
    is_outlier: pd.Series
    outlier_count: int


def _empty_result(method: str, index: pd.Index) -> OutlierResult:
    return OutlierResult(method, pd.Series(False, index=index), 0)


def detect_zscore_outliers(
    df: pd.DataFrame, numeric_columns: list[str], threshold: float = Z_SCORE_THRESHOLD
) -> OutlierResult:
    """Marca como outlier toda fila con |z| > threshold en alguna columna numerica."""
    if not numeric_columns:
        return _empty_result("zscore", df.index)

    subset = df[numeric_columns]
    std = subset.std(ddof=0).replace(0, np.nan)
    z_scores = (subset - subset.mean()) / std
    is_outlier = (z_scores.abs() > threshold).any(axis=1).fillna(False)
    return OutlierResult("zscore", is_outlier, int(is_outlier.sum()))


def detect_iqr_outliers(
    df: pd.DataFrame, numeric_columns: list[str], k: float = IQR_MULTIPLIER
) -> OutlierResult:
    """Marca como outlier toda fila fuera de [Q1-k*IQR, Q3+k*IQR] en alguna columna numerica."""
    if not numeric_columns:
        return _empty_result("iqr", df.index)

    subset = df[numeric_columns]
    q1 = subset.quantile(0.25)
    q3 = subset.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    is_outlier = ((subset < lower) | (subset > upper)).any(axis=1).fillna(False)
    return OutlierResult("iqr", is_outlier, int(is_outlier.sum()))


def detect_isolation_forest_outliers(
    df: pd.DataFrame,
    numeric_columns: list[str],
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> OutlierResult:
    """Detecta outliers multivariados con Isolation Forest sobre columnas numericas estandarizadas."""
    if not numeric_columns:
        return _empty_result("isolation_forest", df.index)

    subset = df[numeric_columns].dropna()
    if len(subset) < 2:
        return _empty_result("isolation_forest", df.index)

    scaled = StandardScaler().fit_transform(subset)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(scaled)  # -1 = outlier, 1 = normal

    is_outlier = pd.Series(False, index=df.index)
    is_outlier.loc[subset.index] = predictions == -1
    return OutlierResult("isolation_forest", is_outlier, int(is_outlier.sum()))


def compare_outlier_methods(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, OutlierResult]:
    """Ejecuta los tres metodos sobre las mismas columnas para poder compararlos (RF-07)."""
    return {
        "zscore": detect_zscore_outliers(df, numeric_columns),
        "iqr": detect_iqr_outliers(df, numeric_columns),
        "isolation_forest": detect_isolation_forest_outliers(df, numeric_columns),
    }


def summarize_outlier_results(results: dict[str, OutlierResult], total_rows: int) -> pd.DataFrame:
    """Tabla comparativa: cantidad y porcentaje de outliers detectados por metodo."""
    rows = {
        name: {
            "outliers_detectados": result.outlier_count,
            "porcentaje": (result.outlier_count / total_rows) if total_rows else 0.0,
        }
        for name, result in results.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")
