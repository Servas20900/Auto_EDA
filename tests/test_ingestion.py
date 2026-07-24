import io

import pandas as pd
import pytest

from modules.ingestion import IngestionError, load_dataset, load_datasets


def _csv_buffer(content: str) -> io.BytesIO:
    return io.BytesIO(content.encode("utf-8"))


class _FakeUploadedFile(io.BytesIO):
    """Simula el objeto que entrega st.file_uploader (tiene .name y .size)."""

    def __init__(self, content: str, name: str):
        super().__init__(content.encode("utf-8"))
        self.name = name
        self.size = len(content.encode("utf-8"))


def test_load_dataset_valid_csv():
    result = load_dataset(_csv_buffer("a,b\n1,x\n2,y\n"), "data.csv")
    assert result.name == "data.csv"
    assert list(result.dataframe.columns) == ["a", "b"]
    assert len(result.dataframe) == 2
    assert result.warnings == []


def test_load_dataset_valid_excel():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    result = load_dataset(buffer, "data.xlsx")
    assert list(result.dataframe.columns) == ["a", "b"]
    assert len(result.dataframe) == 2


def test_load_dataset_unsupported_extension():
    with pytest.raises(IngestionError):
        load_dataset(_csv_buffer("a,b\n1,2\n"), "data.txt")


def test_load_dataset_empty_file():
    with pytest.raises(IngestionError):
        load_dataset(_csv_buffer(""), "empty.csv")


def test_load_dataset_size_limit():
    with pytest.raises(IngestionError):
        load_dataset(_csv_buffer("a,b\n1,2\n"), "big.csv", size_bytes=60 * 1024 * 1024)


def test_load_dataset_warns_on_empty_column():
    result = load_dataset(_csv_buffer("a,b\n1,\n2,\n"), "data.csv")
    assert any("vacias" in w for w in result.warnings)


def test_load_dataset_renames_duplicate_columns():
    # pandas desambigua encabezados repetidos al parsear (a, a.1, ...).
    result = load_dataset(_csv_buffer("a,a\n1,2\n3,4\n"), "data.csv")
    assert list(result.dataframe.columns) == ["a", "a.1"]


def test_load_datasets_mixed_success_and_failure():
    good = _FakeUploadedFile("a,b\n1,2\n", "good.csv")
    bad = _FakeUploadedFile("a,b\n1,2\n", "bad.txt")

    datasets, errors = load_datasets([good, bad])

    assert "good.csv" in datasets
    assert len(errors) == 1
