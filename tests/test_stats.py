import pandas as pd

from modules.stats import CorrelationMethod, compute_descriptive_stats, correlation_matrix
from modules.type_detection import detect_types


def test_numeric_summary_has_expected_columns_and_values():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    result = compute_descriptive_stats(df, detect_types(df))
    assert list(result.numeric.columns) == ["count", "mean", "median", "std", "min", "q1", "q3", "max"]
    assert result.numeric.loc["a", "mean"] == 3.0
    assert result.numeric.loc["a", "min"] == 1
    assert result.numeric.loc["a", "max"] == 5


def test_categorical_summary_reports_mode():
    df = pd.DataFrame({"color": ["rojo", "rojo", "azul"]})
    result = compute_descriptive_stats(df, detect_types(df))
    assert result.categorical.loc["color", "top"] == "rojo"
    assert result.categorical.loc["color", "freq"] == 2
    assert result.categorical.loc["color", "unique"] == 2


def test_boolean_summary_counts_true_false():
    df = pd.DataFrame({"activo": [True, True, False]})
    result = compute_descriptive_stats(df, detect_types(df))
    assert result.boolean.loc["activo", "true_count"] == 2
    assert result.boolean.loc["activo", "false_count"] == 1


def test_temporal_summary_reports_range():
    df = pd.DataFrame({"fecha": ["2024-01-01", "2024-01-10"]})
    result = compute_descriptive_stats(df, detect_types(df))
    assert result.temporal.loc["fecha", "range_days"] == 9


def test_empty_type_group_returns_empty_dataframe_with_expected_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = compute_descriptive_stats(df, detect_types(df))
    assert result.categorical.empty
    assert list(result.categorical.columns) == ["count", "unique", "top", "freq"]


def test_correlation_matrix_pearson_perfect_linear_relation():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    matrix = correlation_matrix(df, ["a", "b"], CorrelationMethod.PEARSON)
    assert matrix.loc["a", "b"] == 1.0


def test_correlation_matrix_requires_at_least_two_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert correlation_matrix(df, ["a"]).empty


def test_correlation_matrix_spearman_perfect_monotonic_relation():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 4, 9, 16, 25]})
    matrix = correlation_matrix(df, ["a", "b"], CorrelationMethod.SPEARMAN)
    assert matrix.loc["a", "b"] == 1.0
