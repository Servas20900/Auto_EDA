"""Estadisticas descriptivas y matrices de correlacion.

Cubre RF-05 (estadisticas descriptivas para variables numericas, categoricas,
booleanas y temporales) y RF-06 (matriz de correlacion Pearson/Spearman).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from modules.type_detection import ColumnClassification, ColumnType, columns_by_type


class CorrelationMethod(str, Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"


@dataclass
class DescriptiveStats:
    """Resumen descriptivo, una tabla por tipo de columna detectado."""

    numeric: pd.DataFrame
    categorical: pd.DataFrame
    boolean: pd.DataFrame
    temporal: pd.DataFrame


def _numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["count", "mean", "median", "std", "min", "q1", "q3", "max"])
    subset = df[columns]
    return pd.DataFrame({
        "count": subset.count(),
        "mean": subset.mean(),
        "median": subset.median(),
        "std": subset.std(),
        "min": subset.min(),
        "q1": subset.quantile(0.25),
        "q3": subset.quantile(0.75),
        "max": subset.max(),
    })


def _categorical_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["count", "unique", "top", "freq"])
    rows = {}
    for column in columns:
        series = df[column].dropna()
        counts = series.value_counts()
        rows[column] = {
            "count": int(series.count()),
            "unique": int(series.nunique()),
            "top": counts.index[0] if not counts.empty else None,
            "freq": int(counts.iloc[0]) if not counts.empty else 0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _boolean_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["count", "true_count", "false_count", "true_ratio"])
    rows = {}
    for column in columns:
        series = df[column].dropna()
        true_count = int((series == True).sum())  # noqa: E712 - comparacion explicita, dtype puede ser object
        false_count = int((series == False).sum())  # noqa: E712
        total = true_count + false_count
        rows[column] = {
            "count": int(series.count()),
            "true_count": true_count,
            "false_count": false_count,
            "true_ratio": (true_count / total) if total else 0.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _temporal_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(columns=["count", "min", "max", "range_days"])
    rows = {}
    for column in columns:
        series = pd.to_datetime(df[column], errors="coerce").dropna()
        rows[column] = {
            "count": int(series.count()),
            "min": series.min() if not series.empty else None,
            "max": series.max() if not series.empty else None,
            "range_days": (series.max() - series.min()).days if not series.empty else None,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def compute_descriptive_stats(df: pd.DataFrame, classifications: list[ColumnClassification]) -> DescriptiveStats:
    """Calcula un resumen descriptivo separado por tipo de columna detectado."""
    return DescriptiveStats(
        numeric=_numeric_summary(df, columns_by_type(classifications, ColumnType.NUMERIC)),
        categorical=_categorical_summary(df, columns_by_type(classifications, ColumnType.CATEGORICAL)),
        boolean=_boolean_summary(df, columns_by_type(classifications, ColumnType.BOOLEAN)),
        temporal=_temporal_summary(df, columns_by_type(classifications, ColumnType.TEMPORAL)),
    )


def correlation_matrix(
    df: pd.DataFrame, numeric_columns: list[str], method: CorrelationMethod = CorrelationMethod.PEARSON
) -> pd.DataFrame:
    """Matriz de correlacion (Pearson o Spearman) entre columnas numericas."""
    if len(numeric_columns) < 2:
        return pd.DataFrame()
    return df[numeric_columns].corr(method=method.value)
