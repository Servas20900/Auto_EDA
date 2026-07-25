import pandas as pd

from modules.outliers import (
    compare_outlier_methods,
    detect_iqr_outliers,
    detect_isolation_forest_outliers,
    detect_zscore_outliers,
    summarize_outlier_results,
)


def _dataset_with_one_outlier() -> pd.DataFrame:
    # Con muestras chicas, el z-score maximo alcanzable esta acotado por
    # (n-1)/sqrt(n): con solo 10 puntos nunca supera ~2.85 aunque el outlier
    # sea enorme. Se usan 30 puntos para que el outlier si cruce el umbral.
    normal_values = [10] * 29
    return pd.DataFrame({"valor": normal_values + [1000]})


def test_detect_zscore_outliers_flags_extreme_value():
    result = detect_zscore_outliers(_dataset_with_one_outlier(), ["valor"])
    assert result.is_outlier.iloc[-1]
    assert result.outlier_count == 1


def test_detect_iqr_outliers_flags_extreme_value():
    result = detect_iqr_outliers(_dataset_with_one_outlier(), ["valor"])
    assert result.is_outlier.iloc[-1]
    assert result.outlier_count >= 1


def test_detect_isolation_forest_outliers_flags_extreme_value():
    result = detect_isolation_forest_outliers(_dataset_with_one_outlier(), ["valor"], contamination=0.1)
    assert result.is_outlier.iloc[-1]


def test_outlier_detection_with_no_numeric_columns_returns_empty():
    df = pd.DataFrame({"color": ["rojo", "azul"]})
    result = detect_zscore_outliers(df, [])
    assert result.outlier_count == 0
    assert not result.is_outlier.any()


def test_isolation_forest_with_insufficient_rows_returns_empty():
    df = pd.DataFrame({"valor": [1.0]})
    result = detect_isolation_forest_outliers(df, ["valor"])
    assert result.outlier_count == 0


def test_compare_outlier_methods_returns_all_three():
    results = compare_outlier_methods(_dataset_with_one_outlier(), ["valor"])
    assert set(results.keys()) == {"zscore", "iqr", "isolation_forest"}


def test_summarize_outlier_results_computes_percentage():
    df = _dataset_with_one_outlier()
    results = compare_outlier_methods(df, ["valor"])
    summary = summarize_outlier_results(results, total_rows=len(df))
    assert summary.loc["zscore", "outliers_detectados"] == 1
    assert abs(summary.loc["zscore", "porcentaje"] - (1 / 30)) < 1e-9
