import pandas as pd
import plotly.graph_objects as go

from modules.report_export import build_html_report


def _minimal_report(**overrides) -> str:
    defaults = dict(
        dataset_name="d",
        cleaning_summary_df=pd.DataFrame(),
        descriptive_stats={},
        correlation_matrix_df=pd.DataFrame(),
        correlation_heatmap=go.Figure(),
        outlier_summary_df=pd.DataFrame(),
        cluster_summary_df=None,
        insights=[],
    )
    defaults.update(overrides)
    return build_html_report(**defaults)


def test_build_html_report_contains_expected_sections():
    stats = {"Variables numericas": pd.DataFrame({"a": [1, 2]})}
    html = _minimal_report(
        dataset_name="mi_dataset.csv",
        descriptive_stats=stats,
        insights=["Frase de ejemplo."],
    )

    assert "mi_dataset.csv" in html
    assert "Frase de ejemplo." in html
    assert "<!doctype html>" in html.lower()
    assert "plotly" in html.lower()


def test_build_html_report_escapes_malicious_dataset_name():
    html = _minimal_report(dataset_name="<script>alert(1)</script>")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_build_html_report_escapes_malicious_column_names():
    stats = {"Variables numericas": pd.DataFrame({"<img src=x onerror=alert(1)>": [1]})}
    html = _minimal_report(descriptive_stats=stats)

    assert "<img src=x onerror=alert(1)>" not in html


def test_build_html_report_with_empty_insights_shows_placeholder():
    html = _minimal_report()
    assert "No se generaron observaciones" in html


def test_build_html_report_includes_cluster_section_when_provided():
    cluster_df = pd.DataFrame({"n_filas": [5, 5]}, index=[0, 1])
    html = _minimal_report(cluster_summary_df=cluster_df)
    assert "Segmentacion" in html


def test_build_html_report_omits_cluster_section_when_not_provided():
    html = _minimal_report(cluster_summary_df=None)
    assert "Segmentacion" not in html
