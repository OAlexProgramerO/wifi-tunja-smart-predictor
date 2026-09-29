# WiFi Tunja Smart Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)](https://scikit-learn.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)](https://streamlit.io/) [![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC)](https://pytest.org/) [![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Leer en [English](README.md)

Aplicación integral de portafolio para clasificar demanda WiFi simulada como **LOW** o **HIGH**. Demuestra un flujo reproducible de datos, evaluación temporal que evita fugas, un pipeline de scikit-learn, una API FastAPI y un dashboard Streamlit.

> **Solo datos sintéticos.** Los puntos de acceso, coordenadas, demanda, clima, eventos, métricas de red y valores históricos son simulados. No representan uso real de WiFi ni infraestructura municipal en Tunja.

## 1. Descripción del proyecto

El proyecto demuestra cómo estructurar un clasificador horario como software mantenible, desde la generación y validación hasta la evaluación y el uso mediante API o dashboard.

## 2. Funcionalidades

- Generador determinista y validación estructurada.
- Variables históricas con semántica temporal explícita.
- Particiones cronológicas de entrenamiento, validación y prueba.
- Cinco modelos base y pipeline persistido.
- API tipada, dashboard multipágina, pruebas pytest, Ruff y Black.

## 3. Problema

Clasificar el nivel esperado para una hora de predicción. `demand_level` es el objetivo principal (`LOW`/`HIGH`); esto no es un servicio de producción.

## 4. Arquitectura

```text
CSV sintético → validación/deduplicación → partición temporal → pipeline sklearn
                                                           ↙               ↘
                                                    API FastAPI       Streamlit
```

La lógica reutilizable está en `src/wifi_tunja_smart_predictor/`. Consulte [arquitectura](docs/architecture.md).

## 5. Dataset

`data/raw/wifi_tunja_public.csv` contiene aproximadamente 60.000 filas, 55 columnas, 30 puntos de acceso y 12 zonas sintéticas en 2024–2025. Incluye valores faltantes esperados y duplicados exactos.

## 6. Aviso sobre los datos

**El dataset es sintético y fue generado para desarrollo de software, experimentación de machine learning, demostración y propósitos de portafolio.** Todas las observaciones, coordenadas, clima, eventos, red y valores históricos son simulados.

## 7. Flujo de ML

Preparación valida, elimina duplicados solo de la copia procesada y genera variables. El pipeline combina ingeniería, `ColumnTransformer` y clasificador; el preprocesamiento se ajusta solo con entrenamiento.

## 8. Prevención de fugas

La marca temporal es el instante de predicción. Las variables históricas preceden ese instante y las métricas de red resumen el periodo anterior. `demand_level` y `connections_next_hour` nunca son entradas. Las particiones son temporales.

## 9. Modelos

Regresión Logística, Árbol de Decisión, Random Forest, K vecinos y Naive Bayes Gaussiano. La selección usa F1 de `HIGH` en validación; la prueba no decide el modelo. Comparación: `reports/metrics/model_comparison.csv`.

## 10. Evaluación

Se reportan exactitud, precisión, exhaustividad, F1, ROC-AUC y matriz de confusión. **La evaluación se basa en datos sintéticos.** Consulte [métricas](reports/metrics/selected_model_test_metrics.json) y [metadatos](models/model_metadata.json); no representan precisión real.

## 11. Dashboard

Ejecute `streamlit run app/dashboard.py`. Incluye resumen, exploración, análisis geográfico y de red, rendimiento, predicción y Acerca de. No presenta telemetría en vivo.

## 12. API

Ejecute `uvicorn api.main:app --reload`; Swagger: [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Endpoints: `GET /health`, `GET /model-info`, `POST /predict`. Consulte [contratos](docs/api.md).

## 13. Estructura

`api/` API; `app/` dashboard; `data/` datasets; `docs/` documentación; `models/` artefactos; `notebooks/` análisis; `reports/` resultados; `scripts/` flujo de trabajo; `src/` paquete; `tests/` pruebas.

## 14. Instalación

Requiere Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## 15. Generar datos

`python scripts/generate_synthetic_dataset.py` genera el CSV sintético de origen.

## 16. Preparar datos

`python scripts/prepare_data.py` valida y escribe los archivos procesados sin modificar el origen.

## 17. Entrenar

`python scripts/train_model.py`

## 18. Evaluar

`python scripts/evaluate_model.py`

## 19. Ejecutar la API

`uvicorn api.main:app --reload`

## 20. Ejecutar el dashboard

`streamlit run app/dashboard.py`

## 21. Pruebas y calidad

`pytest -q` · `ruff check .` · `black --check .`

## 22. Limitaciones

Los datos y resultados son sintéticos. No hay ingestión en vivo, conexión municipal, despliegue ni monitoreo. Las probabilidades no se declaran calibradas. No se implementa regresión.

## 23. Mejoras futuras

Datasets reales autorizados, regresión para conteos futuros, despliegue, monitoreo, almacenamiento, ingestión y explicabilidad son extensiones posibles, fuera de este prototipo.

## 24. Autor

Creado por [OAlexProgramerO](https://github.com/OAlexProgramerO). [Repositorio](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor).
