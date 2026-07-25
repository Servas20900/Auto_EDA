import numpy as np
import pandas as pd

from modules.clustering import (
    DEFAULT_DBSCAN_EPS,
    cluster_summary,
    run_dbscan,
    run_kmeans,
    select_k,
    suggest_dbscan_eps,
)
from modules import clustering


def _two_blobs_df() -> pd.DataFrame:
    # Dos grupos bien separados, como en el criterio de aceptacion del
    # levantamiento de requisitos ("dataset con dos grupos claramente
    # separables -> K-means/DBSCAN los separa correctamente").
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0, 0], scale=0.5, size=(20, 2))
    cluster_b = rng.normal(loc=[10, 10], scale=0.5, size=(20, 2))
    data = np.vstack([cluster_a, cluster_b])
    return pd.DataFrame(data, columns=["x", "y"])


def test_select_k_prefers_two_clusters_for_two_blobs():
    selection = select_k(_two_blobs_df(), ["x", "y"], k_range=range(2, 6))
    assert selection.best_k_silhouette == 2


def test_run_kmeans_separates_two_blobs():
    df = _two_blobs_df()
    result = run_kmeans(df, ["x", "y"], k=2)

    assert result.n_clusters == 2
    first_group = result.labels.iloc[:20]
    second_group = result.labels.iloc[20:]
    assert first_group.nunique() == 1
    assert second_group.nunique() == 1
    assert first_group.iloc[0] != second_group.iloc[0]


def test_run_dbscan_separates_two_blobs_without_noise():
    # Con datos sin ruido real no hay una "rodilla" clara en las distancias,
    # asi que se usa un eps explicito razonable en vez del sugerido, para
    # aislar el comportamiento de DBSCAN del de la heuristica de sugerencia.
    df = _two_blobs_df()
    result = run_dbscan(df, ["x", "y"], eps=0.5, min_samples=3)

    assert result.n_clusters == 2
    assert (result.labels != -1).all()


def test_suggest_dbscan_eps_is_large_enough_to_avoid_flagging_clean_data_as_noise():
    df = _two_blobs_df()
    eps = suggest_dbscan_eps(df, ["x", "y"], min_samples=3)

    result = run_dbscan(df, ["x", "y"], eps=eps, min_samples=3)

    assert result.n_clusters == 2
    assert (result.labels != -1).all()


def test_cluster_summary_reports_size_and_mean_per_cluster():
    df = _two_blobs_df()
    result = run_kmeans(df, ["x", "y"], k=2)

    summary = cluster_summary(df, result.labels, ["x", "y"])

    assert summary["n_filas"].sum() == 40
    assert set(summary.index) == set(result.labels.unique())


def test_select_k_with_insufficient_rows_returns_default():
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    selection = select_k(df, ["x", "y"])
    assert selection.k_values == []
    assert selection.best_k_elbow == 2
    assert selection.best_k_silhouette == 2


def test_suggest_dbscan_eps_with_small_dataset_returns_default():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    eps = suggest_dbscan_eps(df, ["x", "y"], min_samples=5)
    assert eps == DEFAULT_DBSCAN_EPS


def test_run_dbscan_samples_large_datasets_for_performance(monkeypatch):
    # RNF-01: DBSCAN escala peor que K-means; se limita a una muestra en
    # datasets grandes. Se baja el limite a 10 filas para probarlo sin tener
    # que generar un dataset real de mas de 100k filas en el test.
    monkeypatch.setattr(clustering, "DBSCAN_MAX_ROWS", 10)
    df = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float)})

    result = run_dbscan(df, ["x", "y"], eps=1.0, min_samples=2)

    assert result.params["sampled"] is True
    assert result.params["sample_size"] == 10
    assert (result.labels == -1).sum() >= 10  # al menos las filas fuera de la muestra
