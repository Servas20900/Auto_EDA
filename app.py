"""Punto de entrada Streamlit: orquesta ingesta, deteccion de tipos y limpieza.

Fase 1 del plan de desarrollo: carga de archivos, deteccion/ajuste de tipos
y limpieza basica, con vista previa antes y despues.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from modules.cleaning import CleaningOptions, NullStrategy, clean_dataset
from modules.ingestion import load_datasets
from modules.type_detection import ColumnClassification, ColumnType, detect_types

st.set_page_config(page_title="Analisis Exploratorio Automatizado", layout="wide")


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


def _render_dataset(dataset_name: str, df: pd.DataFrame, load_warnings: list[str]) -> None:
    st.subheader(dataset_name)
    st.write(f"{df.shape[0]} filas x {df.shape[1]} columnas")

    for warning in load_warnings:
        st.warning(warning)

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

    if "cleaned_datasets" not in st.session_state:
        st.session_state["cleaned_datasets"] = {}
    st.session_state["cleaned_datasets"][dataset_name] = cleaned_df


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
