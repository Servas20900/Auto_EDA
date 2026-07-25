import pandas as pd

from modules.relationships import detect_target_column, feature_importance_vs_target, top_correlated_pairs
from modules.stats import CorrelationMethod, correlation_matrix
from modules.type_detection import ColumnClassification, ColumnType, detect_types


def test_top_correlated_pairs_orders_by_absolute_value():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],  # perfectamente correlacionada con a
        "c": [5, 4, 3, 2, 1],   # perfectamente anticorrelacionada con a
        "d": [3, 1, 4, 1, 5],   # poco correlacionada
    })
    matrix = correlation_matrix(df, ["a", "b", "c", "d"], CorrelationMethod.PEARSON)

    pairs = top_correlated_pairs(matrix, top_n=3)

    assert abs(pairs.iloc[0]["correlacion"]) == 1.0
    assert abs(pairs.iloc[1]["correlacion"]) == 1.0


def test_top_correlated_pairs_with_empty_matrix():
    result = top_correlated_pairs(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == ["variable_1", "variable_2", "correlacion"]


def test_detect_target_column_by_name_hint():
    df = pd.DataFrame({"edad": [1, 2], "target": [0, 1]})
    assert detect_target_column(detect_types(df)) == "target"


def test_detect_target_column_returns_none_without_hint():
    df = pd.DataFrame({"edad": [1, 2], "ingreso": [100, 200]})
    assert detect_target_column(detect_types(df)) is None


def test_detect_target_column_does_not_confuse_y_coordinate_with_target():
    # Regresion: "y" es un nombre de columna legitimo y muy comun (p.ej. una
    # coordenada), no deberia opacar a una columna 'target' real presente en
    # el mismo dataset.
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "target": [0, 1]})
    assert detect_target_column(detect_types(df)) == "target"


def test_feature_importance_vs_numeric_target_ranks_stronger_predictor_first():
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6, 7, 8],
        "fuerte": [1.1, 1.9, 3.05, 3.95, 5.1, 5.9, 7.05, 7.95],  # casi identica al target
        "debil": [3, 1, 4, 1, 5, 9, 2, 6],  # ruido
    })
    classifications = [
        ColumnClassification("target", ColumnType.NUMERIC, 1.0, ""),
        ColumnClassification("fuerte", ColumnType.NUMERIC, 1.0, ""),
        ColumnClassification("debil", ColumnType.NUMERIC, 1.0, ""),
    ]

    result = feature_importance_vs_target(df, classifications, "target")

    assert result.iloc[0]["variable"] == "fuerte"


def test_feature_importance_vs_target_with_no_other_numeric_columns():
    df = pd.DataFrame({"target": [1, 2, 3], "color": ["a", "b", "c"]})
    classifications = [
        ColumnClassification("target", ColumnType.NUMERIC, 1.0, ""),
        ColumnClassification("color", ColumnType.CATEGORICAL, 1.0, ""),
    ]

    result = feature_importance_vs_target(df, classifications, "target")

    assert result.empty
