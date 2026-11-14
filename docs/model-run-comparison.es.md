# Comparación de ejecuciones de modelos

## Alcance y registros de origen

`GET /model-runs/comparison` lee las tablas persistidas de evaluación de candidatos del clasificador y regresor, junto con los metadatos del modelo seleccionado. Compara candidatos únicamente dentro de la misma tarea: clasificación con clasificación y regresión con regresión. No ajusta modelos ni escribe en archivos de informes o modelos.

El repositorio conserva las tablas de candidatos más recientes, no un historial acumulativo de entrenamientos. Cada fila representa la evaluación de un candidato en ese ciclo guardado, no una secuencia de despliegues independientes. La respuesta informa versión del modelo/paquete, fecha de entrenamiento, versión y huella del dataset cuando están disponibles, huella del artefacto seleccionado y ventanas temporales compartidas. Los candidatos no seleccionados no tienen una huella de artefacto persistida. La evidencia incompleta reduce el estado a `PARTIAL`; una huella o ventana de evaluación incompatible produce `INCOMPATIBLE` y omite las diferencias de métricas.

## Estados

- **VALID** — los registros pertenecen a la misma tarea y coinciden en huella del dataset y ventanas de evaluación; las diferencias disponibles son comparables dentro de esta evaluación sintética.
- **PARTIAL** — faltan algunos registros o campos de procedencia/métricas; solo se muestran los detalles disponibles.
- **INCOMPATIBLE** — difieren la identidad del dataset o las ventanas; se omiten las diferencias de métricas.
- **UNAVAILABLE** — no se pueden leer las tablas o metadatos guardados.

Las diferencias se calculan como candidato menos el seleccionado. En general, valores mayores son favorables para accuracy, precision, recall, F1, ROC-AUC y R²; valores menores para MAE, RMSE y tiempo de entrenamiento. Son resultados descriptivos basados solo en datos sintéticos, no evidencia de desempeño para la demanda real de WiFi. Esta función no lee registros oficiales MinTIC.

Solo se conservan las tablas de candidatos más recientes; esta versión no agrega un archivo histórico de ejecuciones.
