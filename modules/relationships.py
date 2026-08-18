"""Relaciones bivariadas y multivariadas relevantes entre variables.

identifica los pares de columnas mas correlacionados y, cuando
el dataset parece tener una variable objetivo, la importancia de cada
variable numerica sobre ese objetivo (F-test: ANOVA si el objetivo es
categorico/booleano, regresion si es numerico).
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.feature_selection import f_classif, f_regression

from modules.type_detection import ColumnClassification, ColumnType, columns_by_type

TARGET_NAME_HINTS = {
    "target", "label", "labels", "class", "clase", "outcome", "objetivo",
    "resultado", "churn", "default",
}
# Nota: se descarta "y" a proposito. Es una convencion comun para la
# variable objetivo, pero tambien un nombre de columna legitimo y frecuente
# (p.ej. una coordenada), lo que produce falsos positivos.


def top_correlated_pairs(correlation_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Pares de columnas mas correlacionados (en valor absoluto), a partir de una matriz ya calculada."""
    columns = ["variable_1", "variable_2", "correlacion"]
    if correlation_df.empty:
        return pd.DataFrame(columns=columns)

    pairs = [
        {"variable_1": col_a, "variable_2": col_b, "correlacion": correlation_df.loc[col_a, col_b]}
        for col_a, col_b in combinations(correlation_df.columns, 2)
        if pd.notna(correlation_df.loc[col_a, col_b])
    ]
    if not pairs:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(pairs)
    result = result.reindex(result["correlacion"].abs().sort_values(ascending=False).index)
    return result.head(top_n).reset_index(drop=True)


def detect_target_column(classifications: list[ColumnClassification]) -> str | None:
    """Heuristica por nombre para sugerir una posible variable objetivo (RF-09, caso opcional)."""
    for classification in classifications:
        if classification.column.strip().lower() in TARGET_NAME_HINTS:
            return classification.column
    return None


def feature_importance_vs_target(
    df: pd.DataFrame, classifications: list[ColumnClassification], target_column: str
) -> pd.DataFrame:
    """Importancia (F-test) de cada variable numerica sobre `target_column`."""
    columns = ["variable", "importancia", "p_valor"]
    numeric_columns = [c for c in columns_by_type(classifications, ColumnType.NUMERIC) if c != target_column]
    if not numeric_columns:
        return pd.DataFrame(columns=columns)

    data = df[numeric_columns + [target_column]].dropna()
    if len(data) < 3:
        return pd.DataFrame(columns=columns)

    target_type = next((c.detected_type for c in classifications if c.column == target_column), None)
    features, target = data[numeric_columns], data[target_column]

    scores, p_values = f_regression(features, target) if target_type == ColumnType.NUMERIC else f_classif(features, target)

    result = pd.DataFrame({"variable": numeric_columns, "importancia": scores, "p_valor": p_values})
    return result.sort_values("importancia", ascending=False).reset_index(drop=True)
