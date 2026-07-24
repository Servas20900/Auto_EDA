import pandas as pd

from modules.cleaning import CleaningOptions, NullStrategy, clean_dataset
from modules.type_detection import ColumnClassification, ColumnType, detect_types


def test_clean_dataset_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    classifications = detect_types(df)

    cleaned, report = clean_dataset(df, classifications, CleaningOptions(null_strategy=NullStrategy.KEEP))

    assert report.duplicates_removed == 1
    assert report.rows_after == 2
    assert len(cleaned) == 2


def test_clean_dataset_imputes_numeric_with_median():
    df = pd.DataFrame({"a": [1.0, 3.0, None]})
    classifications = detect_types(df)

    cleaned, report = clean_dataset(
        df, classifications,
        CleaningOptions(remove_duplicates=False, null_strategy=NullStrategy.IMPUTE),
    )

    assert cleaned["a"].isna().sum() == 0
    assert cleaned["a"].iloc[2] == 2.0


def test_clean_dataset_imputes_categorical_with_mode():
    df = pd.DataFrame({"color": ["rojo", "rojo", None]})
    classifications = detect_types(df)

    cleaned, report = clean_dataset(
        df, classifications,
        CleaningOptions(remove_duplicates=False, coerce_types=False, null_strategy=NullStrategy.IMPUTE),
    )

    assert cleaned["color"].isna().sum() == 0
    assert cleaned["color"].iloc[2] == "rojo"


def test_clean_dataset_drop_rows_strategy():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    classifications = detect_types(df)

    cleaned, report = clean_dataset(
        df, classifications,
        CleaningOptions(remove_duplicates=False, null_strategy=NullStrategy.DROP_ROWS),
    )

    assert report.rows_after == 2
    assert cleaned["a"].isna().sum() == 0


def test_clean_dataset_keep_strategy_leaves_nulls():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    classifications = detect_types(df)

    cleaned, report = clean_dataset(
        df, classifications,
        CleaningOptions(remove_duplicates=False, null_strategy=NullStrategy.KEEP),
    )

    assert cleaned["a"].isna().sum() == 1
    assert report.rows_after == 3


def test_clean_dataset_reports_coerced_columns():
    df = pd.DataFrame({"a": ["1", "2", "no-es-numero"]})
    # Se fuerza el tipo numerico manualmente, simulando un ajuste del usuario (RF-03).
    forced = [ColumnClassification("a", ColumnType.NUMERIC, 1.0, "forzado")]

    cleaned, report = clean_dataset(
        df, forced, CleaningOptions(remove_duplicates=False, null_strategy=NullStrategy.KEEP),
    )

    assert "a" in report.coerced_columns
    assert cleaned["a"].isna().sum() == 1
