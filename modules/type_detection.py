"""Deteccion automatica del tipo de cada columna de un DataFrame.

Cubre RF-02 (clasificacion automatica: numerica, categorica, temporal,
booleana) y sirve de base para RF-03, ya que cada resultado incluye una
razon legible que la UI puede mostrar antes de permitir un ajuste manual.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

BOOLEAN_TOKENS = {"true", "false", "1", "0", "yes", "no", "si", "verdadero", "falso", "t", "f"}
PARSE_SUCCESS_THRESHOLD = 0.9

_BOOLEAN_MAP = {
    "true": True, "1": True, "yes": True, "si": True, "verdadero": True, "t": True,
    "false": False, "0": False, "no": False, "falso": False, "f": False,
}


class ColumnType(str, Enum):
    NUMERIC = "numerica"
    CATEGORICAL = "categorica"
    TEMPORAL = "temporal"
    BOOLEAN = "booleana"


@dataclass
class ColumnClassification:
    """Tipo detectado para una columna, con confianza y motivo legible."""

    column: str
    detected_type: ColumnType
    confidence: float
    reason: str


def _non_null(series: pd.Series) -> pd.Series:
    return series.dropna()


def _is_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    values = _non_null(series)
    if values.empty:
        return False
    uniques = {str(v).strip().lower() for v in values.unique()}
    # Se exige exactamente 2 valores (no <=2): una columna numerica constante
    # cuyo unico valor sea "0" o "1" (p.ej. un mes fijo) no debe leerse como
    # booleana solo por coincidir con un token valido.
    return len(uniques) == 2 and uniques.issubset(BOOLEAN_TOKENS)


def _temporal_success_ratio(series: pd.Series) -> float:
    values = _non_null(series)
    if values.empty:
        return 0.0
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0
    if pd.api.types.is_numeric_dtype(series):
        # Evita interpretar enteros comunes (edades, conteos) como epoch/fechas.
        return 0.0
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _numeric_success_ratio(series: pd.Series) -> float:
    values = _non_null(series)
    if values.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(series):
        return 1.0
    parsed = pd.to_numeric(values, errors="coerce")
    return float(parsed.notna().mean())


def classify_column(series: pd.Series) -> ColumnClassification:
    """Determina el tipo mas probable de una columna, en orden booleana > temporal > numerica > categorica."""
    name = str(series.name) if series.name is not None else "columna"

    if _is_boolean(series):
        return ColumnClassification(name, ColumnType.BOOLEAN, 1.0, "Solo contiene dos valores booleanos.")

    temporal_ratio = _temporal_success_ratio(series)
    if temporal_ratio >= PARSE_SUCCESS_THRESHOLD:
        return ColumnClassification(
            name, ColumnType.TEMPORAL, temporal_ratio,
            f"{temporal_ratio:.0%} de los valores se interpretan como fecha/hora.",
        )

    numeric_ratio = _numeric_success_ratio(series)
    if numeric_ratio >= PARSE_SUCCESS_THRESHOLD:
        return ColumnClassification(
            name, ColumnType.NUMERIC, numeric_ratio,
            f"{numeric_ratio:.0%} de los valores son numericos.",
        )

    non_null = _non_null(series)
    unique_count = non_null.nunique()
    unique_ratio = unique_count / len(non_null) if len(non_null) else 0.0
    return ColumnClassification(
        name, ColumnType.CATEGORICAL, 1.0 - unique_ratio,
        f"Valores no numericos ni temporales, con {unique_count} categorias distintas.",
    )


def columns_by_type(classifications: list[ColumnClassification], target: ColumnType) -> list[str]:
    """Devuelve los nombres de columna cuyo tipo detectado coincide con `target`."""
    return [c.column for c in classifications if c.detected_type == target]


def detect_types(df: pd.DataFrame) -> list[ColumnClassification]:
    """Clasifica todas las columnas de un DataFrame."""
    return [classify_column(df[col]) for col in df.columns]


def apply_type_overrides(df: pd.DataFrame, overrides: dict[str, ColumnType]) -> pd.DataFrame:
    """Convierte columnas al tipo indicado manualmente por el usuario (RF-03)."""
    result = df.copy()
    for column, target_type in overrides.items():
        if column not in result.columns:
            continue
        if target_type == ColumnType.NUMERIC:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif target_type == ColumnType.TEMPORAL:
            result[column] = pd.to_datetime(result[column], errors="coerce", format="mixed")
        elif target_type == ColumnType.BOOLEAN:
            normalized = result[column].astype(str).str.strip().str.lower()
            result[column] = normalized.map(_BOOLEAN_MAP)
        else:
            # astype(str) directo convertiria los nulos en el texto "nan"; se
            # castea solo la parte no nula, sobre una copia object, para
            # preservarlos como NaN sin disparar el FutureWarning de pandas
            # por asignar strings en una columna numerica.
            working = result[column].astype(object)
            mask = working.notna()
            working[mask] = working[mask].astype(str)
            result[column] = working
    return result
