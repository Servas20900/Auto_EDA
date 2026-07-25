# Sistema Inteligente para Analisis Automatizado y Exploratorio de Datos

Aplicacion web (Streamlit) que recibe conjuntos de datos tabulares (CSV/Excel)
y genera de forma autonoma un analisis exploratorio: deteccion de tipos,
limpieza, estadisticas, correlaciones, outliers, clustering e insights en
lenguaje natural. Ver el detalle completo en el documento de levantamiento
de requisitos del proyecto.

## Requisitos

- Python 3.11+
- Ver `requirements.txt` para las dependencias exactas.

## Instalacion y ejecucion

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Estructura del proyecto

```
app.py                  punto de entrada Streamlit (UI, orquestacion)
/modules
  ingestion.py          carga de archivos, validacion de formato
  type_detection.py      clasificacion de columnas por tipo
  cleaning.py            limpieza basica (nulos, duplicados, tipos)
  stats.py               estadisticas descriptivas y correlaciones
  outliers.py            Z-score, IQR, Isolation Forest
  clustering.py          K-means, DBSCAN, seleccion de k
  relationships.py       relaciones bivariadas/multivariadas
  visualization.py       graficos con plotly
  insights.py            resumen interpretativo en lenguaje natural
  report_export.py       exportacion del reporte a HTML
/tests
  test_*.py              pruebas unitarias por modulo
requirements.txt
```

## Tests

```bash
pytest
```
