import pandas as pd

from modules.cleaning import CleaningReport
from modules.clustering import ClusteringResult
from modules.insights import (
    build_summary,
    describe_cleaning,
    describe_clusters,
    describe_correlations,
    describe_outliers,
    describe_target_importance,
)
from modules.outliers import OutlierResult


def test_describe_correlations_labels_strong_positive_relationship():
    pairs = pd.DataFrame({"variable_1": ["a"], "variable_2": ["b"], "correlacion": [0.87]})
    result = describe_correlations(pairs)
    assert "fuerte" in result[0]
    assert "positiva" in result[0]
    assert "0.87" in result[0]


def test_describe_correlations_labels_negative_direction():
    pairs = pd.DataFrame({"variable_1": ["a"], "variable_2": ["b"], "correlacion": [-0.8]})
    result = describe_correlations(pairs)
    assert "negativa" in result[0]


def test_describe_correlations_labels_weak_relationship():
    pairs = pd.DataFrame({"variable_1": ["a"], "variable_2": ["b"], "correlacion": [0.1]})
    result = describe_correlations(pairs)
    assert "debil" in result[0]


def test_describe_correlations_with_empty_pairs_returns_placeholder():
    result = describe_correlations(pd.DataFrame())
    assert len(result) == 1


def test_describe_outliers_reports_each_method():
    results = {
        "zscore": OutlierResult("zscore", pd.Series([False, True]), 1),
        "iqr": OutlierResult("iqr", pd.Series([False, False]), 0),
    }
    insights = describe_outliers(results, total_rows=2)
    assert any("zscore" in text and "1" in text for text in insights)
    assert any("iqr" in text and "no detecto" in text for text in insights)


def test_describe_outliers_with_no_results_returns_placeholder():
    assert len(describe_outliers({}, total_rows=0)) == 1


def test_describe_clusters_reports_noise_for_dbscan():
    labels = pd.Series([0, 0, -1, 1])
    result = ClusteringResult("dbscan", labels, n_clusters=2, params={})
    insights = describe_clusters(result, total_rows=4)
    assert "2 clusters" in insights[0]
    assert "ruido" in insights[0]


def test_describe_clusters_kmeans_has_no_noise_mention():
    labels = pd.Series([0, 0, 1, 1])
    result = ClusteringResult("kmeans", labels, n_clusters=2, params={})
    insights = describe_clusters(result, total_rows=4)
    assert "ruido" not in insights[0]


def test_describe_cleaning_reports_duplicates_and_nulls_and_coerced_columns():
    report = CleaningReport(
        rows_before=10, rows_after=8, duplicates_removed=2,
        nulls_before={"a": 3}, nulls_after={"a": 0}, coerced_columns=["a"],
    )
    insights = describe_cleaning(report)
    assert any("2 filas duplicadas" in text for text in insights)
    assert any("3 valores nulos" in text for text in insights)
    assert any("'a'" in text for text in insights)


def test_describe_cleaning_with_nothing_to_report_returns_empty_list():
    report = CleaningReport(rows_before=5, rows_after=5, duplicates_removed=0, nulls_before={}, nulls_after={}, coerced_columns=[])
    assert describe_cleaning(report) == []


def test_describe_target_importance_reports_top_variable():
    importance = pd.DataFrame({"variable": ["x"], "importancia": [12.5], "p_valor": [0.001]})
    insights = describe_target_importance(importance, "target")
    assert "'x'" in insights[0]
    assert "'target'" in insights[0]


def test_describe_target_importance_with_empty_dataframe_returns_empty_list():
    assert describe_target_importance(pd.DataFrame(), "target") == []


def test_build_summary_combines_all_sections():
    report = CleaningReport(10, 10, 0, {}, {}, [])
    pairs = pd.DataFrame({"variable_1": ["a"], "variable_2": ["b"], "correlacion": [0.9]})
    outlier_results = {"zscore": OutlierResult("zscore", pd.Series([False] * 10), 0)}

    summary = build_summary(report, pairs, outlier_results, total_rows=10)

    assert any("fuerte" in text for text in summary)
    assert any("zscore" in text for text in summary)
