"""Limpieza basica y configurable de un dataset ya clasificado por tipos.

Cubre RF-04: manejo de nulos, duplicados y tipos inconsistentes, con
estrategias que la UI puede exponer como opciones al usuario.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from modules.type_detection import ColumnClassification, ColumnType, apply_type_overrides


class NullStrategy(str, Enum):
    KEEP = "mantener"
    DROP_ROWS = "eliminar_filas"
    IMPUTE = "imputar"


@dataclass
class CleaningOptions:
    """Configuracion de limpieza elegida por el usuario en la UI."""

    remove_duplicates: bool = True
    null_strategy: NullStrategy = NullStrategy.IMPUTE
    coerce_types: bool = True


@dataclass
class CleaningReport:
    """Resumen de las transformaciones aplicadas, pensado para mostrarse en la UI."""

    rows_before: int
    rows_after: int
    duplicates_removed: int
    nulls_before: dict[str, int]
    nulls_after: dict[str, int]
    coerced_columns: list[str] = field(default_factory=list)


def _coerce_inconsistent_types(
    df: pd.DataFrame, classifications: list[ColumnClassification]
) -> tuple[pd.DataFrame, list[str]]:
    """Convierte cada columna a su tipo detectado; los valores que no encajan quedan como nulos."""
    overrides = {c.column: c.detected_type for c in classifications}
    before_nulls = df.isna().sum()
    result = apply_type_overrides(df, overrides)
    after_nulls = result.isna().sum()
    coerced = [col for col in result.columns if after_nulls.get(col, 0) > before_nulls.get(col, 0)]
    return result, coerced


def _impute(df: pd.DataFrame, classifications: list[ColumnClassification]) -> pd.DataFrame:
    """Imputa nulos: mediana para numericas, moda para categoricas/booleanas."""
    result = df.copy()
    type_by_column = {c.column: c.detected_type for c in classifications}
    for column in result.columns:
        if result[column].isna().sum() == 0:
            continue
        column_type = type_by_column.get(column)
        if column_type == ColumnType.NUMERIC:
            result[column] = result[column].fillna(result[column].median())
        elif column_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
            mode = result[column].mode(dropna=True)
            if not mode.empty:
                result[column] = result[column].fillna(mode.iloc[0])
        # Las columnas temporales no se imputan: no existe un valor neutro razonable.
    return result


def clean_dataset(
    df: pd.DataFrame,
    classifications: list[ColumnClassification],
    options: CleaningOptions | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Aplica coercion de tipos, duplicados y manejo de nulos segun `options`."""
    options = options or CleaningOptions()
    rows_before = len(df)
    nulls_before = df.isna().sum().to_dict()

    result = df
    coerced_columns: list[str] = []
    if options.coerce_types:
        result, coerced_columns = _coerce_inconsistent_types(result, classifications)

    duplicates_removed = 0
    if options.remove_duplicates:
        duplicate_mask = result.duplicated()
        duplicates_removed = int(duplicate_mask.sum())
        result = result.loc[~duplicate_mask].reset_index(drop=True)

    if options.null_strategy == NullStrategy.DROP_ROWS:
        result = result.dropna().reset_index(drop=True)
    elif options.null_strategy == NullStrategy.IMPUTE:
        result = _impute(result, classifications)

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(result),
        duplicates_removed=duplicates_removed,
        nulls_before=nulls_before,
        nulls_after=result.isna().sum().to_dict(),
        coerced_columns=coerced_columns,
    )
    return result, report
