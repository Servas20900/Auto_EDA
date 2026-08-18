"""Generacion de graficos interactivos con plotly.

histogramas, boxplots, scatter plots, heatmap de correlacion
 y scatter/boxplot coloreados por grupo, usados tanto para resaltar
outliers como para el grafico de clusters .
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Histograma de una columna numerica."""
    return px.histogram(df, x=column, title=f"Distribucion de {column}")


def boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Boxplot de una columna numerica."""
    return px.box(df, y=column, title=f"Boxplot de {column}")


def scatter(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Scatter plot entre dos columnas numericas."""
    return px.scatter(df, x=x, y=y, title=f"{y} vs {x}")


def correlation_heatmap(correlation_df: pd.DataFrame) -> go.Figure:
    """Heatmap de una matriz de correlacion ya calculada (RF-06)."""
    if correlation_df.empty:
        return go.Figure()
    return px.imshow(
        correlation_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Matriz de correlacion",
    )


def boxplot_by_group(df: pd.DataFrame, column: str, groups: pd.Series, group_name: str = "grupo") -> go.Figure:
    """Boxplot de una columna coloreado por una serie de grupos (p.ej. outlier si/no, o cluster)."""
    data = df[[column]].copy()
    data[group_name] = groups.reindex(data.index).astype(str)
    return px.box(data, y=column, color=group_name, points="all", title=f"{column} por {group_name}")


def scatter_by_group(df: pd.DataFrame, x: str, y: str, groups: pd.Series, group_name: str = "grupo") -> go.Figure:
    """Scatter entre dos columnas numericas, coloreado por una serie de grupos arbitraria.

    Los valores de `groups` se castean a texto para que plotly los trate
    como categorias discretas (p.ej. ids de cluster) y no como una escala
    continua.
    """
    data = df[[x, y]].copy()
    data[group_name] = groups.reindex(data.index).astype(str)
    return px.scatter(data, x=x, y=y, color=group_name, title=f"{y} vs {x} por {group_name}")


def scatter_with_outliers(df: pd.DataFrame, x: str, y: str, is_outlier: pd.Series) -> go.Figure:
    """Scatter entre dos columnas numericas, resaltando las filas marcadas como outliers."""
    groups = is_outlier.fillna(False).map({True: "atipico", False: "normal"})
    return scatter_by_group(df, x, y, groups, group_name="estado")
