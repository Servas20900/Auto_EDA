import pandas as pd

from modules.type_detection import ColumnType, apply_type_overrides, classify_column, detect_types


def test_classify_numeric_column():
    series = pd.Series([1, 2, 3, 4], name="edad")
    assert classify_column(series).detected_type == ColumnType.NUMERIC


def test_classify_numeric_column_stored_as_strings():
    series = pd.Series(["1", "2", "3.5"], name="monto")
    assert classify_column(series).detected_type == ColumnType.NUMERIC


def test_classify_boolean_column():
    series = pd.Series(["yes", "no", "yes"], name="activo")
    assert classify_column(series).detected_type == ColumnType.BOOLEAN


def test_classify_temporal_column():
    series = pd.Series(["2024-01-01", "2024-02-01", "2024-03-15"], name="fecha")
    assert classify_column(series).detected_type == ColumnType.TEMPORAL


def test_classify_categorical_column():
    series = pd.Series(["rojo", "verde", "azul", "rojo"], name="color")
    assert classify_column(series).detected_type == ColumnType.CATEGORICAL


def test_integer_column_not_treated_as_temporal():
    series = pd.Series([2020, 2021, 2022], name="anio")
    assert classify_column(series).detected_type == ColumnType.NUMERIC


def test_constant_numeric_column_with_value_one_not_treated_as_boolean():
    # Regresion: una columna como MONTH_NUM cuando el dataset solo cubre un
    # mes queda con el unico valor 1 repetido, y no debe leerse como booleana.
    series = pd.Series([1, 1, 1, 1], name="month_num")
    assert classify_column(series).detected_type == ColumnType.NUMERIC


def test_binary_numeric_column_with_both_values_treated_as_boolean():
    series = pd.Series([0, 1, 1, 0], name="is_active")
    assert classify_column(series).detected_type == ColumnType.BOOLEAN


def test_detect_types_returns_one_result_per_column_in_order():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    results = detect_types(df)
    assert [r.column for r in results] == ["a", "b"]


def test_apply_type_overrides_numeric_to_categorical_preserves_nulls():
    df = pd.DataFrame({"a": [1.0, 2.0, None]})
    overridden = apply_type_overrides(df, {"a": ColumnType.CATEGORICAL})
    assert overridden["a"].isna().sum() == 1
    assert overridden["a"].dropna().tolist() == ["1.0", "2.0"]


def test_apply_type_overrides_to_boolean():
    df = pd.DataFrame({"flag": ["yes", "no", "yes"]})
    overridden = apply_type_overrides(df, {"flag": ColumnType.BOOLEAN})
    assert overridden["flag"].tolist() == [True, False, True]


def test_apply_type_overrides_to_temporal():
    df = pd.DataFrame({"fecha": ["2024-01-01", "no-es-fecha"]})
    overridden = apply_type_overrides(df, {"fecha": ColumnType.TEMPORAL})
    assert overridden["fecha"].notna().sum() == 1
