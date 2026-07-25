"""Generacion de texto interpretativo a partir de resultados ya calculados.

Cubre RF-11: reglas basadas en umbrales que traducen resultados numericos
(limpieza, correlaciones, outliers, clusters, importancia de variables) en
frases en lenguaje natural para el resumen del dashboard y del reporte.
"""

from __future__ import annotations

import pandas as pd

from modules.cleaning import CleaningReport
from modules.clustering import ClusteringResult
from modules.outliers import OutlierResult

STRONG_CORRELATION_THRESHOLD = 0.7
MODERATE_CORRELATION_THRESHOLD = 0.4


def _correlation_strength_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= STRONG_CORRELATION_THRESHOLD:
        return "fuerte"
    if abs_value >= MODERATE_CORRELATION_THRESHOLD:
        return "moderada"
    return "debil"


def describe_correlations(pairs: pd.DataFrame, max_items: int = 3) -> list[str]:
    """Frases sobre los pares de variables mas correlacionados (RF-06, RF-09)."""
    if pairs.empty:
        return ["No se encontraron suficientes variables numericas para evaluar correlaciones."]

    insights = []
    for _, row in pairs.head(max_items).iterrows():
        strength = _correlation_strength_label(row["correlacion"])
        direction = "positiva" if row["correlacion"] >= 0 else "negativa"
        insights.append(
            f"Existe una correlacion {strength} {direction} (r={row['correlacion']:.2f}) "
            f"entre '{row['variable_1']}' y '{row['variable_2']}'."
        )
    return insights


def describe_outliers(results: dict[str, OutlierResult], total_rows: int) -> list[str]:
    """Frases sobre los outliers detectados por cada metodo (RF-07)."""
    if not results or total_rows == 0:
        return ["No se evaluaron outliers (no hay columnas numericas)."]

    insights = []
    for name, result in results.items():
        if result.outlier_count == 0:
            insights.append(f"El metodo {name} no detecto valores atipicos.")
        else:
            pct = result.outlier_count / total_rows
            insights.append(f"El metodo {name} detecto {result.outlier_count} valores atipicos ({pct:.1%} de las filas).")
    return insights


def describe_clusters(result: ClusteringResult, total_rows: int) -> list[str]:
    """Frase sobre el resultado de clustering, K-means o DBSCAN (RF-08)."""
    if result.n_clusters == 0:
        return [f"El metodo {result.method} no encontro clusters diferenciados."]

    noise = int((result.labels == -1).sum()) if result.method == "dbscan" else 0
    sentence = f"El metodo {result.method} identifico {result.n_clusters} clusters"
    if noise and total_rows:
        sentence += f", dejando un {noise / total_rows:.1%} de las filas como ruido/no clasificado"
    return [sentence + "."]


def describe_cleaning(report: CleaningReport) -> list[str]:
    """Frases sobre la limpieza aplicada al dataset (RF-04)."""
    insights = []
    if report.duplicates_removed:
        insights.append(f"Se eliminaron {report.duplicates_removed} filas duplicadas.")

    total_nulls_before = sum(report.nulls_before.values())
    if total_nulls_before:
        total_nulls_after = sum(report.nulls_after.values())
        insights.append(
            f"El dataset original tenia {total_nulls_before} valores nulos; "
            f"quedaron {total_nulls_after} despues de la limpieza."
        )

    if report.coerced_columns:
        columns = ", ".join(f"'{c}'" for c in report.coerced_columns)
        insights.append(f"Se corrigieron valores con tipos inconsistentes en: {columns}.")

    return insights


def describe_target_importance(importance: pd.DataFrame, target_column: str) -> list[str]:
    """Frase sobre la variable con mayor relacion con el objetivo elegido (RF-09)."""
    if importance.empty:
        return []
    top = importance.iloc[0]
    return [
        f"La variable con mayor relacion con '{target_column}' es '{top['variable']}' "
        f"(F={top['importancia']:.2f}, p={top['p_valor']:.4f})."
    ]


def build_summary(
    cleaning_report: CleaningReport,
    correlation_pairs: pd.DataFrame,
    outlier_results: dict[str, OutlierResult],
    total_rows: int,
    clustering_result: ClusteringResult | None = None,
    target_importance: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> list[str]:
    """Arma la lista completa de frases interpretativas del dataset (RF-11)."""
    insights: list[str] = []
    insights += describe_cleaning(cleaning_report)
    insights += describe_correlations(correlation_pairs)
    insights += describe_outliers(outlier_results, total_rows)
    if clustering_result is not None:
        insights += describe_clusters(clustering_result, total_rows)
    if target_importance is not None and target_column is not None:
        insights += describe_target_importance(target_importance, target_column)
    return insights
