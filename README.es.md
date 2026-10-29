# WiFi Tunja Smart Predictor — V6.4.0

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)](https://www.python.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)](https://scikit-learn.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B)](https://streamlit.io/) [![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC)](https://pytest.org/) [![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Leer en [English](README.md)

**Estado de publicación:** V6.4.0 está publicada con la etiqueta Git anotada [`v6.4.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.4.0) y [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.4.0). Commit de publicación: `8841996993aec18405f19e143796175cafa51540`; paquete Python e imagen Docker: `0.6.4`. Consulte el [registro de validación V6.4.0](docs/releases/v6.4.0.md), la [guía de diagnósticos del modelo](docs/model-diagnostics.md) y el [historial de versiones](docs/release-history.md).

WiFi Tunja Smart Predictor es una demostración con datos sintéticos de clasificación horaria de demanda y estimación del número de conexiones, disponible mediante FastAPI, un dashboard Streamlit y un asistente determinista. V6.4.0 presenta con mayor claridad el resumen de evaluación existente en Model Performance, sin cambiar los modelos ni su metodología de evaluación.

Se conservan las verificaciones de huellas de artefactos de V4.3.0 y la trazabilidad de predicciones y validaciones de V4.2.0. Consulte el [diseño de diagnósticos V4.4.0](docs/model-diagnostics.md) para ver el esquema, las métricas, las ventanas temporales y la comprobación de procedencia.

El proyecto usa únicamente datos sintéticos. Sus métricas no demuestran rendimiento predictivo para la demanda real de WiFi público ni para infraestructura municipal en Tunja. Consulte la [metodología](docs/methodology.md), [limitaciones](docs/limitations.md), [guía de API](docs/api.md) y [guía del asistente](docs/assistant.md).

**Límite de seguridad:** Las protecciones de API son una base práctica. No se incluyen autenticación, autorización, limitación de velocidad para producción, WAF ni despliegue probado mediante pentesting.

> **Solo datos sintéticos.** Los puntos de acceso, coordenadas, demanda, clima, eventos, métricas de red y valores históricos son simulados. No representan uso real de WiFi ni infraestructura municipal en Tunja.

## Inicio rápido con Docker

Requisito: instale e inicie Docker Desktop (o Docker Engine con Compose). La demostración en contenedores no requiere instalar Python en el host, crear un entorno virtual, generar los datos ni entrenar los modelos manualmente.

```powershell
git clone https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor.git
cd wifi-tunja-smart-predictor
docker compose up --build --wait --wait-timeout 900
```

En el primer inicio, `runtime-init` genera el dataset sintético, entrena los dos modelos existentes y los evalúa para crear el informe de diagnóstico en volúmenes administrados por Compose. Espere a que las API y el dashboard estén disponibles. Abra el dashboard en <http://localhost:8501>, la documentación de la API principal en <http://localhost:8000/docs> y la del asistente en <http://localhost:8001/docs>. Revise `docker compose ps`, siga los registros con `docker compose logs -f`, ejecute las pruebas HTTP desde el contenedor del dashboard con `docker compose exec -T dashboard python scripts/smoke_test.py` y detenga con `docker compose down`. Los volúmenes de datos/modelos persisten después de `down`; `docker compose down -v` elimina solo los volúmenes generados de este proyecto.

El registro V5.3.0 incluye verificaciones remotas aprobadas de compilación y arranque de Docker, smoke HTTP y persistencia; esta máquina local no tenía un daemon Docker activo. Consulte el [registro de validación V5.3.0](docs/releases/v5.3.0.md) para ver los resultados y el alcance de la auditoría.

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
- Registros JSON de ciclo de solicitud y eventos operativos correlacionados con `X-Request-ID`, sin guardar mensajes ni cuerpos de solicitud.
- Protecciones básicas de API: límites de solicitudes, CORS controlado, encabezados de seguridad, errores de validación saneados y contexto restringido.
- Conversación básica en inglés y español para saludos, identidad y capacidades, sin distinguir mayúsculas ni puntuación externa.
- Consultas naturales de demanda con zonas, horas y contexto de ubicación reutilizado en preguntas posteriores.
- Explicaciones deterministas del asistente para las secciones del panel, escenarios actuales, indicadores y métricas, en inglés y español.
- Análisis histórico de demanda en inglés y español: comparación de zonas, frecuencia HIGH, horas/días pico y periodos del día.
- Diez secciones: resumen, escenario, explorador, análisis geográfico y de red, rendimiento, historial de experimentos, asistente, predicción avanzada y acerca de.
- Experimentos reproducibles de planificación de capacidad, persistidos en SQLite, con asignación determinista, historial paginado e informes JSON/Markdown. La página «Historial de experimentos» está conectada a los escenarios guardados; las métricas son conexiones simuladas, no rendimiento real.
- Inteligencia geoespacial con filtros por fechas inclusivas, zona, hora, día y clase de demanda sobre observaciones históricas generadas, y mapa sintético sin mosaicos externos.
- Banco de escenarios que compara predicciones fijas del modelo con supuestos alternativos de capacidad y presupuesto, guarda experimentos reproducibles y exporta informes con el análisis.
- El banco también permite optimizar asignaciones enteras deterministas para minimizar déficits sintéticos totales o ponderados explícitamente; consulte [método y restricciones](docs/optimizacion.md).
- V6.2.0 agrega análisis acotado de sensibilidad sintética de demanda por zona y barridos de presupuesto sobre optimizaciones guardadas; consulte la [guía de sensibilidad y robustez](docs/robustness.es.md). Estos casos seleccionados no estiman probabilidades ni confiabilidad real.
- V6.3.0 agrega una API de evaluación temporal acotada y una vista en Model Performance con informes versionados JSON/Markdown; consulte la [guía](docs/temporal-evaluation.es.md). Los resultados usan solo datos sintéticos.

## Inicio rápido de desarrollo local

Para desarrollo local, instale con `python -m pip install -e ".[dev]"`, entrene ambos modelos con `python scripts/train_model.py` y evalúe con `python scripts/evaluate_model.py`. Inicie el dashboard con `streamlit run app/dashboard.py`, la API principal con `uvicorn api.main:app --reload --port 8000` y el asistente con `uvicorn assistant_api.main:app --reload --port 8001`. `pyproject.toml` es la fuente única de dependencias; los dos archivos `requirements` son entradas de compatibilidad. El desarrollo local admite Python 3.11 o posterior; CI y el contenedor usan Python 3.12. Las dependencias de ejecución bloqueadas con hashes se dirigen a Linux x86-64; Compose selecciona `linux/amd64` para permitir emulación en Docker Desktop sobre equipos ARM.

Consulte [Predicción de escenarios](docs/scenario_prediction.md), [Asistente](docs/assistant.md), [Rendimiento](docs/performance.md), [Geoespacial](docs/geospatial.md), [Seguridad](docs/security.md) y [Limitaciones, privacidad y seguridad](docs/limitations.md).

La API incluye `POST /evaluation/temporal`, `POST /experiments/scenarios`, `POST /experiments/what-if`, V6.1 `POST /experiments/optimize`, V6.2.0 `POST /experiments/robustness`, `GET /experiments` con paginación, `GET /experiments/{id}` y `GET /experiments/{id}/report` en JSON o Markdown (puerto 8000). Consulte [referencia API en español](docs/api.es.md), [evaluación temporal](docs/temporal-evaluation.es.md), [planificación de capacidad](docs/capacity-planning.md), [experimentos e informes](docs/experiments.md), el [banco de escenarios](docs/scenario-workbench.md), [optimización](docs/optimizacion.md) y [sensibilidad y robustez](docs/robustness.es.md).

Para guardar un experimento, abra **Live Scenario**, elija una zona y hora sintéticas, ejecute la predicción, indique la capacidad simulada y un presupuesto opcional de conexiones y pulse **Run and save experiment**. **Experiment History** muestra los escenarios guardados y permite descargar el informe JSON versionado. Para comparar, obtenga dos IDs mediante `GET /experiments` y envíelos en `POST /experiments/compare`; se exige igualdad de periodo, huella del dataset y huellas de artefactos de modelo. Estas simulaciones no son recomendaciones de configuración para redes reales.

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

Model Performance presenta el [informe de diagnóstico versionado](docs/model-diagnostics.md) con versión del modelo, fecha de evaluación, métricas principales de clasificación y regresión y el aviso de datos sintéticos. Cada ventana cronológica usa solo filas sintéticas anteriores; los resultados no validan demanda pública real de WiFi.

## 11. Dashboard

Ejecute `streamlit run app/dashboard.py`. Incluye diez secciones, escenarios por ubicación/hora, mapa de AP sintéticos, historial de experimentos y asistente determinista. No presenta telemetría en vivo.

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
python -m pip install -e ".[dev]"
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
