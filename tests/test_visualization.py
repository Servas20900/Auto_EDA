import pandas as pd
import plotly.graph_objects as go

from modules.visualization import (
    boxplot,
    boxplot_by_group,
    correlation_heatmap,
    histogram,
    scatter,
    scatter_by_group,
    scatter_with_outliers,
)


def test_histogram_returns_figure():
    df = pd.DataFrame({"a": [1, 2, 2, 3]})
    fig = histogram(df, "a")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_boxplot_returns_figure():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    fig = boxplot(df, "a")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_scatter_returns_figure_with_expected_points():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    fig = scatter(df, "a", "b")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 3


def test_correlation_heatmap_with_empty_matrix_returns_empty_figure():
    fig = correlation_heatmap(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_correlation_heatmap_with_matrix_returns_figure():
    matrix = pd.DataFrame({"a": [1.0, 0.5], "b": [0.5, 1.0]}, index=["a", "b"])
    fig = correlation_heatmap(matrix)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_boxplot_by_group_colors_all_points():
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    groups = pd.Series(["normal", "normal", "atipico", "normal"])
    fig = boxplot_by_group(df, "a", groups, group_name="estado")
    assert isinstance(fig, go.Figure)
    total_points = sum(len(trace.y) for trace in fig.data)
    assert total_points == 4


def test_scatter_by_group_supports_more_than_two_groups():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    groups = pd.Series([0, 1, 1, -1])
    fig = scatter_by_group(df, "a", "b", groups, group_name="cluster")
    assert isinstance(fig, go.Figure)
    total_points = sum(len(trace.x) for trace in fig.data)
    assert total_points == 4
    assert len(fig.data) == 3  # una serie por cada valor distinto de cluster


def test_scatter_with_outliers_colors_all_points():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    is_outlier = pd.Series([False, False, True])
    fig = scatter_with_outliers(df, "a", "b", is_outlier)
    assert isinstance(fig, go.Figure)
    total_points = sum(len(trace.x) for trace in fig.data)
    assert total_points == 3
