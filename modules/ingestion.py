"""Carga y validacion de archivos CSV/Excel subidos por el usuario.

Cubre RF-01 (carga de uno o varios archivos) y RNF-03 (manejo controlado
de errores: archivo corrupto, vacio o con formato no soportado).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 50


class IngestionError(Exception):
    """Error esperado durante la carga de un archivo (mensaje apto para el usuario)."""


@dataclass
class LoadedDataset:
    """Resultado de cargar un archivo: nombre original, DataFrame y advertencias."""

    name: str
    dataframe: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _check_size(size_bytes: int, filename: str) -> None:
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise IngestionError(
            f"'{filename}' supera el tamano maximo permitido ({MAX_FILE_SIZE_MB} MB)."
        )


def load_csv(buffer: Any, filename: str) -> pd.DataFrame:
    """Lee un buffer como CSV, traduciendo errores comunes a IngestionError."""
    try:
        return pd.read_csv(buffer)
    except pd.errors.EmptyDataError as exc:
        raise IngestionError(f"'{filename}' esta vacio o no contiene datos.") from exc
    except UnicodeDecodeError as exc:
        raise IngestionError(
            f"'{filename}' tiene una codificacion no soportada. Guardalo en UTF-8 e intenta de nuevo."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - se traduce a mensaje claro para el usuario
        raise IngestionError(f"'{filename}' no pudo ser interpretado como CSV: {exc}") from exc


def load_excel(buffer: Any, filename: str) -> pd.DataFrame:
    """Lee un buffer como Excel, traduciendo errores comunes a IngestionError."""
    try:
        return pd.read_excel(buffer)
    except Exception as exc:  # noqa: BLE001 - se traduce a mensaje claro para el usuario
        raise IngestionError(
            f"'{filename}' no pudo ser interpretado como Excel (posible archivo corrupto): {exc}"
        ) from exc


def load_dataset(file_obj: Any, filename: str, size_bytes: int | None = None) -> LoadedDataset:
    """Valida y carga un unico archivo, devolviendo el DataFrame y advertencias no bloqueantes."""
    extension = _get_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        formatos = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestionError(
            f"'{filename}' tiene un formato no soportado ({extension or 'sin extension'}). "
            f"Formatos aceptados: {formatos}."
        )

    if size_bytes is not None:
        _check_size(size_bytes, filename)

    dataframe = load_csv(file_obj, filename) if extension == ".csv" else load_excel(file_obj, filename)

    if dataframe.empty:
        raise IngestionError(f"'{filename}' no contiene filas de datos.")

    warnings: list[str] = []
    empty_columns = [col for col in dataframe.columns if dataframe[col].isna().all()]
    if empty_columns:
        warnings.append(f"Columnas completamente vacias: {empty_columns}.")

    return LoadedDataset(name=filename, dataframe=dataframe, warnings=warnings)


def load_datasets(uploaded_files: list[Any]) -> tuple[dict[str, LoadedDataset], list[str]]:
    """Carga varios archivos subidos.

    Un archivo que falle no detiene la carga del resto: su mensaje de error se
    agrega a la lista de errores para que la UI lo muestre sin bloquear los
    archivos validos (RF-01, RNF-03).
    """
    datasets: dict[str, LoadedDataset] = {}
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", str(uploaded_file))
        try:
            size_bytes = getattr(uploaded_file, "size", None)
            datasets[filename] = load_dataset(uploaded_file, filename, size_bytes)
        except IngestionError as exc:
            errors.append(str(exc))

    return datasets, errors
