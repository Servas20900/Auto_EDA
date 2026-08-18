"""Exportacion del reporte de analisis a un archivo HTML autocontenido.

Se elige HTML (en vez de PDF) porque el resultado es un unico
archivo auto-contenido, sin dependencias nativas fragiles (p.ej. weasyprint
necesita librerias del sistema), y plotly.js se embebe inline en vez de via
CDN para que el reporte pueda abrirse sin conexion a internet (RNF-05).
"""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import plotly.graph_objects as go

REPORT_STYLE = """
<style>
  body { font-family: Arial, sans-serif; margin: 2rem; color: #1a1a1a; }
  h1 { border-bottom: 2px solid #333; padding-bottom: .3rem; }
  h2 { margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
  table.data-table { border-collapse: collapse; margin: .5rem 0 1.5rem; font-size: .9rem; }
  table.data-table th, table.data-table td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }
  table.data-table th { background: #f2f2f2; }
  .meta { color: #666; font-size: .85rem; }
  .figure { margin: 1rem 0; }
</style>
"""


def _figure_to_html(fig: go.Figure, include_plotlyjs: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)


def _dataframe_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>Sin datos.</em></p>"
    # escape=True es el default de pandas: tambien escapa nombres de columna,
    # que vienen del archivo subido por el usuario y no son de confianza.
    return df.to_html(classes="data-table", border=0, escape=True)


def _insights_to_html(insights: list[str]) -> str:
    if not insights:
        return "<p><em>No se generaron observaciones automaticas.</em></p>"
    items = "".join(f"<li>{escape(text)}</li>" for text in insights)
    return f"<ul>{items}</ul>"


def build_html_report(
    dataset_name: str,
    cleaning_summary_df: pd.DataFrame,
    descriptive_stats: dict[str, pd.DataFrame],
    correlation_matrix_df: pd.DataFrame,
    correlation_heatmap: go.Figure,
    outlier_summary_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame | None,
    insights: list[str],
) -> str:
    """Arma un reporte HTML autocontenido con los resultados de un dataset (RF-12)."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    safe_name = escape(dataset_name)

    sections = [
        f"<h1>Reporte de analisis exploratorio: {safe_name}</h1>",
        f"<p class='meta'>Generado el {generated_at}</p>",
        "<h2>Resumen interpretativo</h2>",
        _insights_to_html(insights),
        "<h2>Limpieza de datos</h2>",
        _dataframe_to_html(cleaning_summary_df),
        "<h2>Estadisticas descriptivas</h2>",
    ]

    for label, stats_df in descriptive_stats.items():
        if stats_df is not None and not stats_df.empty:
            sections.append(f"<h3>{escape(label)}</h3>")
            sections.append(_dataframe_to_html(stats_df))

    sections += [
        "<h2>Correlacion</h2>",
        _dataframe_to_html(correlation_matrix_df),
        f"<div class='figure'>{_figure_to_html(correlation_heatmap, include_plotlyjs=True)}</div>",
        "<h2>Outliers</h2>",
        _dataframe_to_html(outlier_summary_df),
    ]

    if cluster_summary_df is not None:
        sections += ["<h2>Segmentacion (clusters)</h2>", _dataframe_to_html(cluster_summary_df)]

    body = "\n".join(sections)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Reporte - {safe_name}</title>{REPORT_STYLE}</head><body>{body}</body></html>"
    )
