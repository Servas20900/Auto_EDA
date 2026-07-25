"""Punto de entrada Streamlit: orquesta ingesta, limpieza y analisis exploratorio.

Fase 1: carga de archivos, deteccion/ajuste de tipos y limpieza basica.
Fase 2: estadisticas descriptivas, correlaciones y deteccion de outliers.
Fase 3: clustering (K-means, DBSCAN) y relaciones multivariadas.
Fase 4: resumen interpretativo y exportacion del reporte.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modules import visualization
from modules.cleaning import CleaningOptions, CleaningReport, NullStrategy, clean_dataset
from modules.clustering import (
    DEFAULT_MIN_SAMPLES,
    ClusteringResult,
    KSelection,
    cluster_summary,
    run_dbscan,
    run_kmeans,
    select_k,
    suggest_dbscan_eps,
)
from modules.ingestion import load_datasets
from modules.insights import build_summary
from modules.outliers import OutlierResult, compare_outlier_methods, summarize_outlier_results
from modules.relationships import detect_target_column, feature_importance_vs_target, top_correlated_pairs
from modules.report_export import build_html_report
from modules.stats import CorrelationMethod, DescriptiveStats, compute_descriptive_stats, correlation_matrix
from modules.type_detection import ColumnClassification, ColumnType, columns_by_type, detect_types

st.set_page_config(page_title="Analisis Exploratorio Automatizado", layout="wide")


@st.cache_data(show_spinner="Calculando outliers...")
def _cached_outlier_comparison(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, OutlierResult]:
    return compare_outlier_methods(df, numeric_columns)


@st.cache_data(show_spinner="Explorando valores de k...")
def _cached_k_selection(df: pd.DataFrame, numeric_columns: list[str]) -> KSelection:
    return select_k(df, numeric_columns)


@st.cache_data(show_spinner="Sugiriendo eps para DBSCAN...")
def _cached_suggest_eps(df: pd.DataFrame, numeric_columns: list[str], min_samples: int) -> float:
    return suggest_dbscan_eps(df, numeric_columns, min_samples=min_samples)


def _type_editor(dataset_name: str, classifications: list[ColumnClassification]) -> dict[str, ColumnType]:
    """Muestra la clasificacion automatica y permite ajustarla manualmente (RF-03)."""
    st.caption("Tipo detectado por columna. Ajusta manualmente si es necesario.")
    type_options = [t.value for t in ColumnType]
    overrides: dict[str, ColumnType] = {}

    for classification in classifications:
        col_name, col_type, col_reason = st.columns([2, 2, 3])
        col_name.write(classification.column)
        selected = col_type.selectbox(
            label=classification.column,
            options=type_options,
            index=type_options.index(classification.detected_type.value),
            key=f"type_{dataset_name}_{classification.column}",
            label_visibility="collapsed",
        )
        col_reason.caption(f"{classification.reason} (confianza {classification.confidence:.0%})")
        overrides[classification.column] = ColumnType(selected)

    return overrides


def _cleaning_controls(dataset_name: str) -> CleaningOptions:
    """Controles de limpieza configurables desde la UI (RF-04)."""
    st.caption("Opciones de limpieza")
    col1, col2, col3 = st.columns(3)
    remove_duplicates = col1.checkbox("Eliminar filas duplicadas", value=True, key=f"dedupe_{dataset_name}")
    coerce_types = col2.checkbox("Corregir tipos inconsistentes", value=True, key=f"coerce_{dataset_name}")
    strategy_label = col3.selectbox(
        "Manejo de valores nulos",
        options=[s.value for s in NullStrategy],
        key=f"nulls_{dataset_name}",
    )
    return CleaningOptions(
        remove_duplicates=remove_duplicates,
        coerce_types=coerce_types,
        null_strategy=NullStrategy(strategy_label),
    )


def _apply_overrides(
    classifications: list[ColumnClassification], overrides: dict[str, ColumnType]
) -> list[ColumnClassification]:
    adjusted = []
    for classification in classifications:
        target_type = overrides.get(classification.column, classification.detected_type)
        if target_type == classification.detected_type:
            adjusted.append(classification)
        else:
            adjusted.append(
                ColumnClassification(
                    classification.column, target_type, classification.confidence,
                    "Ajustado manualmente por el usuario.",
                )
            )
    return adjusted


def _render_ingestion_and_cleaning(
    dataset_name: str, df: pd.DataFrame
) -> tuple[pd.DataFrame, list[ColumnClassification], CleaningReport]:
    st.markdown("**Vista previa**")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Deteccion de tipos**")
    classifications = detect_types(df)
    overrides = _type_editor(dataset_name, classifications)
    adjusted_classifications = _apply_overrides(classifications, overrides)

    st.markdown("**Limpieza**")
    cleaning_options = _cleaning_controls(dataset_name)
    cleaned_df, report = clean_dataset(df, adjusted_classifications, cleaning_options)

    st.markdown("**Resultado de la limpieza**")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Filas antes", report.rows_before)
    metric_cols[1].metric("Filas despues", report.rows_after)
    metric_cols[2].metric("Duplicados eliminados", report.duplicates_removed)

    if report.coerced_columns:
        st.info(f"Columnas con valores inconsistentes corregidos: {', '.join(report.coerced_columns)}")

    st.markdown("**Vista previa (datos limpios)**")
    st.dataframe(cleaned_df.head(20), use_container_width=True)

    return cleaned_df, adjusted_classifications, report


def _render_descriptive_stats(df: pd.DataFrame, classifications: list[ColumnClassification]) -> DescriptiveStats:
    stats = compute_descriptive_stats(df, classifications)
    if not stats.numeric.empty:
        st.caption("Variables numericas")
        st.dataframe(stats.numeric, use_container_width=True)
    if not stats.categorical.empty:
        st.caption("Variables categoricas")
        st.dataframe(stats.categorical, use_container_width=True)
    if not stats.boolean.empty:
        st.caption("Variables booleanas")
        st.dataframe(stats.boolean, use_container_width=True)
    if not stats.temporal.empty:
        st.caption("Variables temporales")
        st.dataframe(stats.temporal, use_container_width=True)
    return stats


def _render_correlation(dataset_name: str, df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if len(numeric_columns) < 2:
        st.info("Se necesitan al menos dos columnas numericas para calcular correlaciones.")
        return pd.DataFrame()

    method_label = st.selectbox(
        "Metodo de correlacion", options=[m.value for m in CorrelationMethod], key=f"corr_method_{dataset_name}"
    )
    matrix = correlation_matrix(df, numeric_columns, CorrelationMethod(method_label))
    st.plotly_chart(
        visualization.correlation_heatmap(matrix), use_container_width=True, key=f"corr_heatmap_{dataset_name}"
    )
    st.dataframe(matrix, use_container_width=True)
    return matrix


def _render_relationships(
    dataset_name: str, df: pd.DataFrame, classifications: list[ColumnClassification], correlation_df: pd.DataFrame
) -> pd.DataFrame:
    """Pares mas correlacionados y, si se elige una variable objetivo, importancia de variables (RF-09)."""
    if correlation_df.empty:
        st.info("Se necesitan al menos dos columnas numericas para identificar relaciones relevantes.")
        return pd.DataFrame()

    pairs = top_correlated_pairs(correlation_df, top_n=10)
    st.caption("Pares de variables mas correlacionados")
    st.dataframe(pairs, use_container_width=True)

    suggested_target = detect_target_column(classifications)
    target_options = ["(ninguna)"] + [c.column for c in classifications]
    default_index = target_options.index(suggested_target) if suggested_target in target_options else 0
    target_label = st.selectbox(
        "Variable objetivo (opcional)", options=target_options, index=default_index, key=f"target_{dataset_name}"
    )
    if target_label == "(ninguna)":
        return pairs

    importance = feature_importance_vs_target(df, classifications, target_label)
    if importance.empty:
        st.info("No hay suficientes variables numericas para calcular importancia sobre el objetivo elegido.")
    else:
        st.caption(f"Importancia de variables numericas sobre '{target_label}' (F-test)")
        st.dataframe(importance, use_container_width=True)

    return pairs


def _render_outliers(dataset_name: str, df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, OutlierResult]:
    if not numeric_columns:
        st.info("No hay columnas numericas para detectar outliers.")
        return {}

    results = _cached_outlier_comparison(df, numeric_columns)
    st.caption("Comparacion entre metodos (RF-07)")
    st.dataframe(summarize_outlier_results(results, total_rows=len(df)), use_container_width=True)

    method_label = st.selectbox("Metodo a visualizar", options=list(results.keys()), key=f"outlier_method_{dataset_name}")
    column = st.selectbox("Columna", options=numeric_columns, key=f"outlier_column_{dataset_name}")

    is_outlier = results[method_label].is_outlier
    groups = is_outlier.map({True: "atipico", False: "normal"})
    fig = visualization.boxplot_by_group(df, column, groups, group_name="estado")
    st.plotly_chart(fig, use_container_width=True, key=f"outlier_box_{dataset_name}")

    if len(numeric_columns) >= 2:
        x_col, y_col = numeric_columns[0], numeric_columns[1]
        st.plotly_chart(
            visualization.scatter_with_outliers(df, x_col, y_col, is_outlier),
            use_container_width=True,
            key=f"outlier_scatter_{dataset_name}",
        )

    return results


def _render_charts(dataset_name: str, df: pd.DataFrame, numeric_columns: list[str]) -> None:
    if not numeric_columns:
        st.info("No hay columnas numericas para graficar.")
        return

    col_hist, col_box = st.columns(2)
    hist_column = col_hist.selectbox("Histograma de", options=numeric_columns, key=f"hist_col_{dataset_name}")
    col_hist.plotly_chart(
        visualization.histogram(df, hist_column), use_container_width=True, key=f"hist_{dataset_name}"
    )

    box_column = col_box.selectbox("Boxplot de", options=numeric_columns, key=f"box_col_{dataset_name}")
    col_box.plotly_chart(visualization.boxplot(df, box_column), use_container_width=True, key=f"box_{dataset_name}")

    if len(numeric_columns) >= 2:
        scatter_x = st.selectbox("Eje X", options=numeric_columns, key=f"scatter_x_{dataset_name}")
        remaining = [c for c in numeric_columns if c != scatter_x]
        scatter_y = st.selectbox("Eje Y", options=remaining, key=f"scatter_y_{dataset_name}")
        st.plotly_chart(
            visualization.scatter(df, scatter_x, scatter_y),
            use_container_width=True,
            key=f"scatter_{dataset_name}",
        )


def _render_statistics(dataset_name: str, df: pd.DataFrame, classifications: list[ColumnClassification]) -> dict:
    numeric_columns = columns_by_type(classifications, ColumnType.NUMERIC)

    st.markdown("**Estadisticas descriptivas**")
    descriptive_stats = _render_descriptive_stats(df, classifications)

    st.markdown("**Correlacion**")
    correlation_df = _render_correlation(dataset_name, df, numeric_columns)

    st.markdown("**Relaciones destacadas**")
    correlation_pairs = _render_relationships(dataset_name, df, classifications, correlation_df)

    st.markdown("**Deteccion de outliers**")
    outlier_results = _render_outliers(dataset_name, df, numeric_columns)

    st.markdown("**Graficos**")
    _render_charts(dataset_name, df, numeric_columns)

    return {
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_df,
        "correlation_pairs": correlation_pairs,
        "outlier_results": outlier_results,
    }


def _render_kmeans_section(dataset_name: str, df: pd.DataFrame, numeric_columns: list[str]) -> ClusteringResult | None:
    k_selection = _cached_k_selection(df, numeric_columns)

    if not k_selection.k_values:
        st.info("Muy pocas filas para explorar clustering automaticamente.")
        return None

    if len(k_selection.k_values) == 1:
        k = k_selection.k_values[0]
        st.caption(f"El dataset solo permite probar k={k} (muy pocas filas).")
    else:
        col_inertia, col_silhouette = st.columns(2)
        col_inertia.caption("Inercia por k (metodo del codo)")
        col_inertia.line_chart(pd.Series(k_selection.inertias, index=k_selection.k_values, name="inercia"))
        col_silhouette.caption("Silhouette por k")
        col_silhouette.line_chart(
            pd.Series(k_selection.silhouette_scores, index=k_selection.k_values, name="silhouette")
        )
        st.caption(
            f"Sugerido por metodo del codo: k={k_selection.best_k_elbow}. "
            f"Sugerido por silhouette: k={k_selection.best_k_silhouette}."
        )
        k = st.slider(
            "Numero de clusters (K-means)",
            min_value=min(k_selection.k_values),
            max_value=max(k_selection.k_values),
            value=k_selection.best_k_silhouette,
            key=f"kmeans_k_{dataset_name}",
        )

    return run_kmeans(df, numeric_columns, k)


def _render_dbscan_section(dataset_name: str, df: pd.DataFrame, numeric_columns: list[str]) -> ClusteringResult:
    min_samples = st.number_input(
        "min_samples", min_value=2, value=DEFAULT_MIN_SAMPLES, step=1, key=f"dbscan_min_samples_{dataset_name}"
    )
    suggested_eps = _cached_suggest_eps(df, numeric_columns, int(min_samples))
    eps = st.number_input(
        "eps",
        min_value=0.01,
        value=round(suggested_eps, 3),
        step=0.05,
        key=f"dbscan_eps_{dataset_name}",
        help=f"Sugerido automaticamente a partir de los datos: {suggested_eps:.3f}",
    )

    result = run_dbscan(df, numeric_columns, eps=eps, min_samples=int(min_samples))
    if result.params.get("sampled"):
        st.info(
            f"DBSCAN se calculo sobre una muestra aleatoria de {result.params['sample_size']} filas "
            "por rendimiento (RNF-01); el resto queda marcado como no clasificado (-1)."
        )
    st.caption(f"DBSCAN encontro {result.n_clusters} clusters.")
    return result


def _render_clustering(
    dataset_name: str, df: pd.DataFrame, classifications: list[ColumnClassification]
) -> ClusteringResult | None:
    numeric_columns = columns_by_type(classifications, ColumnType.NUMERIC)
    if len(numeric_columns) < 2:
        st.info("Se necesitan al menos dos columnas numericas para hacer clustering.")
        return None

    st.markdown("**K-means**")
    kmeans_result = _render_kmeans_section(dataset_name, df, numeric_columns)

    st.markdown("**DBSCAN**")
    dbscan_result = _render_dbscan_section(dataset_name, df, numeric_columns)

    available_results = {
        name: result for name, result in [("kmeans", kmeans_result), ("dbscan", dbscan_result)] if result is not None
    }
    if not available_results:
        return None

    method_label = st.radio(
        "Metodo a visualizar", options=list(available_results.keys()), horizontal=True, key=f"cluster_method_{dataset_name}"
    )
    result = available_results[method_label]

    st.markdown("**Resumen por cluster**")
    st.dataframe(cluster_summary(df, result.labels, numeric_columns), use_container_width=True)

    st.markdown("**Grafico de clusters**")
    x_col = st.selectbox("Eje X", options=numeric_columns, key=f"cluster_x_{dataset_name}")
    remaining = [c for c in numeric_columns if c != x_col]
    y_col = st.selectbox("Eje Y", options=remaining, key=f"cluster_y_{dataset_name}")
    st.plotly_chart(
        visualization.scatter_by_group(df, x_col, y_col, result.labels, group_name="cluster"),
        use_container_width=True,
        key=f"cluster_scatter_{dataset_name}",
    )

    return result


def _render_report(
    dataset_name: str,
    df: pd.DataFrame,
    classifications: list[ColumnClassification],
    cleaning_report: CleaningReport,
    stats_bundle: dict,
    clustering_result: ClusteringResult | None,
) -> None:
    """Resumen interpretativo (RF-11) y descarga del reporte HTML (RF-12)."""
    total_rows = len(df)
    correlation_pairs = stats_bundle["correlation_pairs"]
    outlier_results = stats_bundle["outlier_results"]

    target_label = st.session_state.get(f"target_{dataset_name}", "(ninguna)")
    target_column = target_label if target_label not in (None, "(ninguna)") else None
    target_importance = (
        feature_importance_vs_target(df, classifications, target_column) if target_column else None
    )

    insights = build_summary(
        cleaning_report=cleaning_report,
        correlation_pairs=correlation_pairs,
        outlier_results=outlier_results,
        total_rows=total_rows,
        clustering_result=clustering_result,
        target_importance=target_importance,
        target_column=target_column,
    )

    st.markdown("**Resumen interpretativo**")
    if insights:
        for text in insights:
            st.markdown(f"- {text}")
    else:
        st.info("No hay suficientes resultados para generar un resumen.")

    stats = stats_bundle["descriptive_stats"]
    descriptive_stats_dict = {
        "Variables numericas": stats.numeric,
        "Variables categoricas": stats.categorical,
        "Variables booleanas": stats.boolean,
        "Variables temporales": stats.temporal,
    }
    cleaning_summary_df = pd.DataFrame({
        "metrica": ["Filas antes", "Filas despues", "Duplicados eliminados"],
        "valor": [cleaning_report.rows_before, cleaning_report.rows_after, cleaning_report.duplicates_removed],
    })
    outlier_summary_df = (
        summarize_outlier_results(outlier_results, total_rows) if outlier_results else pd.DataFrame()
    )
    numeric_columns = columns_by_type(classifications, ColumnType.NUMERIC)
    cluster_summary_df = (
        cluster_summary(df, clustering_result.labels, numeric_columns) if clustering_result is not None else None
    )

    html_report = build_html_report(
        dataset_name=dataset_name,
        cleaning_summary_df=cleaning_summary_df,
        descriptive_stats=descriptive_stats_dict,
        correlation_matrix_df=stats_bundle["correlation_matrix"],
        correlation_heatmap=visualization.correlation_heatmap(stats_bundle["correlation_matrix"]),
        outlier_summary_df=outlier_summary_df,
        cluster_summary_df=cluster_summary_df,
        insights=insights,
    )

    st.markdown("**Descargar reporte**")
    st.download_button(
        "Descargar reporte (HTML)",
        data=html_report,
        file_name=f"reporte_{Path(dataset_name).stem}.html",
        mime="text/html",
        key=f"download_{dataset_name}",
    )


def _render_dataset(dataset_name: str, df: pd.DataFrame, load_warnings: list[str]) -> None:
    st.subheader(dataset_name)
    st.write(f"{df.shape[0]} filas x {df.shape[1]} columnas")

    for warning in load_warnings:
        st.warning(warning)

    ingestion_tab, stats_tab, clustering_tab, report_tab = st.tabs(
        ["Ingesta y limpieza", "Analisis estadistico", "Segmentacion", "Reporte"]
    )

    with ingestion_tab:
        cleaned_df, classifications, cleaning_report = _render_ingestion_and_cleaning(dataset_name, df)
        if "cleaned_datasets" not in st.session_state:
            st.session_state["cleaned_datasets"] = {}
        st.session_state["cleaned_datasets"][dataset_name] = cleaned_df

    with stats_tab:
        st.caption("Calculado sobre los datos ya limpios de la pestana 'Ingesta y limpieza'.")
        stats_bundle = _render_statistics(dataset_name, cleaned_df, classifications)

    with clustering_tab:
        st.caption("Calculado sobre los datos ya limpios de la pestana 'Ingesta y limpieza'.")
        clustering_result = _render_clustering(dataset_name, cleaned_df, classifications)

    with report_tab:
        _render_report(dataset_name, cleaned_df, classifications, cleaning_report, stats_bundle, clustering_result)


def main() -> None:
    st.title("Sistema Inteligente de Analisis Exploratorio de Datos")
    st.write(
        "Sube uno o varios archivos CSV o Excel para detectar tipos de columna, "
        "aplicar limpieza basica y previsualizar el resultado."
    )

    uploaded_files = st.file_uploader(
        "Archivos CSV o Excel",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Esperando archivos para comenzar el analisis.")
        return

    datasets, errors = load_datasets(uploaded_files)

    for error in errors:
        st.error(error)

    if not datasets:
        return

    tabs = st.tabs(list(datasets.keys()))
    for tab, (name, loaded) in zip(tabs, datasets.items()):
        with tab:
            _render_dataset(name, loaded.dataframe, loaded.warnings)


if __name__ == "__main__":
    main()
