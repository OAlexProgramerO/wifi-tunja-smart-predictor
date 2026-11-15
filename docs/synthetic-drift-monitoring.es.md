# Monitoreo de deriva en datos sintéticos — V8.0.0

## Datos y prevención de filtraciones

La comparación usa únicamente las columnas permitidas de `MODEL_INPUT_COLUMNS`: contrasta la partición cronológica sintética de entrenamiento con una partición `validation` o `test` seleccionada por separado. Las columnas objetivo `demand_level` y `connections_next_hour` nunca se leen como features. El monitor no acepta una lista arbitraria de variables; excluye identificadores y cualquier columna que no esté en la lista permitida del modelo. Las variables históricas son admisibles según la regla del generador: solo utilizan observaciones anteriores al timestamp de predicción de cada fila.

No se usan la base ni las instantáneas oficiales MinTIC. El monitor no carga modelos, no entrena, no cambia artefactos ni modifica predicciones.

## Estadísticas y umbrales

- Las columnas numéricas informan media, mediana, desviación estándar, proporciones faltantes, conteo de valores no numéricos e índice PSI (Population Stability Index). PSI usa hasta diez intervalos por cuantiles calculados en la partición de referencia y un suavizado fijo pequeño para intervalos vacíos.
- Las columnas categóricas informan conteos, distancia de variación total sobre la unión de categorías, proporciones faltantes y categorías que aparecen en evaluación pero no en referencia.
- El mínimo predeterminado es de 20 valores no faltantes en ambas particiones. Con menos muestras el estado es `INSUFFICIENT_SAMPLES` y no se calcula indicador de deriva.
- PSI entre 0,10 y 0,20, cambio medio normalizado de 1–2 desviaciones estándar de referencia, variación total entre 0,10 y 0,25, diferencia de faltantes entre 0,05 y 0,20 o cualquier categoría nueva producen `WARNING`. PSI ≥ 0,20, cambio medio ≥ 2 desviaciones estándar, variación total ≥ 0,25, diferencia de faltantes ≥ 0,20 o proporción de categorías nuevas ≥ 0,10 producen `DRIFT_DETECTED` para esa variable.
- El estado general es `DRIFT_DETECTED` si alguna variable supera su umbral de deriva; en caso contrario es `WARNING` si alguna variable advierte o no tiene suficientes muestras; si no, es `STABLE`.

Si falta una columna aprobada del modelo en alguna partición, aparece como `MISSING_FEATURE` y el informe general es al menos `WARNING`.

Son umbrales descriptivos fijos, no pruebas de hipótesis. PSI y variación total pueden ser sensibles a intervalos/categorías y tamaño de muestra; no se aplica corrección por comparaciones múltiples ni se evalúa desempeño del modelo. Una señal no demuestra degradación, causalidad, cambios de distribución real ni cambios en la demanda pública de WiFi en Tunja.

## API e informes guardados

`GET /drift/monitor?evaluation_split=validation|test` calcula un informe nuevo con el dataset sintético disponible y no escribe archivos. Las huellas de origen y particiones solo calculan hashes de features permitidas del modelo; nunca incluyen valores de objetivos. `scripts/generate_drift_report.py --evaluation-split validation|test` guarda explícitamente un JSON versionado en `reports/metrics/drift/`, con huellas solo de features, fecha de generación, variables, partición, umbrales y configuración de muestras. El archivo es un informe generado, no un artefacto de entrenamiento. Ejecute el script por separado para cada partición.
