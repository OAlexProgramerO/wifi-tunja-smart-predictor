# WiFi Tunja Smart Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)](https://scikit-learn.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)](https://streamlit.io/) [![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC)](https://pytest.org/) [![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Leer en [English](README.md)

**Versión actual: 0.3.1.** V3 es un prototipo de apoyo a decisiones con datos sintéticos. A partir de una ubicación y hora estima demanda LOW/HIGH, conexiones para la próxima hora, intervalo calibrado con validación, uso aproximado de capacidad y sensibilidad local del modelo.

> **Solo datos sintéticos.** Los puntos de acceso, coordenadas, demanda, clima, eventos, métricas de red y valores históricos son simulados. No representan uso real de WiFi ni infraestructura municipal en Tunja.

## 1. Descripción del proyecto

El proyecto demuestra un flujo mantenible de clasificación y regresión con construcción de escenarios por ubicación y hora, evaluación temporal y uso mediante API o dashboard.

## 2. Funcionalidades

- Generador determinista y validación estructurada.
- Variables históricas con semántica temporal explícita.
- Particiones cronológicas de entrenamiento, validación y prueba.
- Cinco clasificadores y dos regresores base, con artefactos separados.
- Constructor determinista de escenarios, resolución de AP sintéticos y contexto histórico; sin telemetría en vivo ni clave de mapas de pago.
- Clasificador y regresor separados, entrenados con las mismas particiones cronológicas.
- Asistente determinista basado en herramientas y API independiente en el puerto 8001; no requiere credenciales LLM.
- Conversación básica en inglés y español para saludos, identidad y capacidades, sin distinguir mayúsculas ni puntuación externa.
- Nueve secciones: resumen, escenario, explorador, análisis geográfico y de red, rendimiento, asistente, predicción avanzada y acerca de.

## Inicio rápido V3

Instale con `python -m pip install -r requirements-dev.txt`, entrene ambos modelos con `python scripts/train_model.py` y evalúe con `python scripts/evaluate_model.py`. Inicie el dashboard con `streamlit run app/dashboard.py`, la API principal con `uvicorn api.main:app --reload --port 8000` y el asistente con `uvicorn assistant_api.main:app --reload --port 8001`.

Consulte [Predicción de escenarios](docs/scenario_prediction.md), [Asistente](docs/assistant.md), [Geoespacial](docs/geospatial.md) y [Limitaciones, privacidad y seguridad](docs/limitations.md).

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

Preparación valida, elimina duplicados solo de la copia procesada y genera variables. Los pipelines combinan ingeniería, `ColumnTransformer` y estimador; el preprocesamiento se ajusta solo con entrenamiento.

## 8. Prevención de fugas

La marca temporal es el instante de predicción. Las variables históricas preceden ese instante y las métricas de red resumen el periodo anterior. `demand_level` y `connections_next_hour` nunca son entradas. Las particiones son temporales.

## 9. Modelos

La clasificación compara cinco modelos y selecciona por F1 de `HIGH` en validación. La regresión compara Random Forest y HistGradientBoosting y selecciona por MAE de validación. Un intervalo split conformal usa residuos de validación; la cobertura de prueba se informa aparte.

## 10. Evaluación

Se reportan exactitud, precisión, exhaustividad, F1, ROC-AUC y matriz de confusión. **La evaluación se basa en datos sintéticos.** Consulte [métricas](reports/metrics/selected_model_test_metrics.json) y [metadatos](models/model_metadata.json); no representan precisión real.

## 11. Dashboard

Ejecute `streamlit run app/dashboard.py`. Incluye nueve secciones, constructor de escenarios por ubicación/hora, mapa interactivo de AP sintéticos y asistente determinista. No presenta telemetría en vivo.

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

Los datos y resultados son sintéticos. No hay ingestión en vivo ni conexión municipal. Los intervalos pueden perder cobertura ante cambios de distribución; la sensibilidad local no es causal. La capacidad usa conexiones como aproximación.

## 23. Mejoras futuras

Datos reales autorizados, proveedores en vivo, despliegue y monitoreo requieren trabajo futuro, fuera de este prototipo.

## 24. Autor

Creado por [OAlexProgramerO](https://github.com/OAlexProgramerO). [Repositorio](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor).
