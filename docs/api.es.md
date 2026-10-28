# Referencia de API — evaluación temporal V6.3.0 y experimentos

La API principal corre en el puerto 8000; Swagger está disponible en `/docs`.

## Evaluación temporal V6.3.0

`POST /evaluation/temporal` ajusta copias de los pipelines de clasificación y regresión en 2–5 ventanas cronológicas expansivas, con un mínimo de 100 filas de entrenamiento y 20 de prueba por defecto. La solicitud puede definir `min_train_rows` (2–50.000), `min_test_rows` (2–10.000) y entre 1 y 12 variables de entrada. La respuesta JSON versionada incluye límites, métricas, comparaciones de distribución sintética, huellas SHA-256 del dataset y artefactos y limitaciones. Use `POST /evaluation/temporal?report_format=markdown` para un resumen legible. Solo son datos sintéticos; las diferencias no prueban deterioro ni deriva real. Consulte la [guía de evaluación temporal](temporal-evaluation.es.md).

## Crear una optimización

`POST /experiments/optimize` valida y guarda un problema de asignación de conexiones enteras para entre una y doce zonas sintéticas. La solicitud incluye `name`, `datetime`, `objective`, `total_budget_connections` (0–100 000), `max_allocation_steps` (1–100 000) y `zones`. Cada zona incluye `zone_id`, `capacity_connections` y, opcionalmente, asignación de referencia y límites mínimos/máximos. El objetivo ponderado `MINIMIZE_WEIGHTED_DEFICIT` exige un `priority_weight` positivo por cada zona; el objetivo `MINIMIZE_TOTAL_DEFICIT` rechaza pesos.

Ejemplo ilustrativo, solo con valores sintéticos:

```json
{
  "name": "Asignación ilustrativa",
  "datetime": "2025-01-01T12:00:00",
  "objective": "MINIMIZE_TOTAL_DEFICIT",
  "total_budget_connections": 3,
  "zones": [
    {"zone_id": "zona_sintetica_a", "capacity_connections": 3},
    {"zone_id": "zona_sintetica_b", "capacity_connections": 4}
  ]
}
```

Una solicitud válida devuelve HTTP 201 y el registro guardado en el mismo almacén SQLite existente. Zona desconocida: 422. Artefactos de modelo no disponibles: 503. Las respuestas conservan el identificador de solicitud `X-Request-ID`; los mensajes de error son sanitizados.

## Recuperar y exportar

V6.2.0 agrega `POST /experiments/robustness` para una optimización guardada. Acepta rangos inclusivos acotados de multiplicador sintético (0,25–2; hasta 6 valores totales incluyendo la referencia 1,0), presupuesto (0–100 000; hasta 20 valores totales incluyendo el presupuesto padre), máximo 100 evaluaciones y máximo 10 000 pasos del optimizador por caso. Modifica una zona a la vez y barre presupuesto por separado; incluye las referencias originales si faltan en los rangos. Separa el puntaje de la asignación original bajo cada supuesto del resultado reoptimizado. La integridad incompleta del padre no se marca como verificada. Esta función se publicó en V6.2.0. Consulte [sensibilidad y robustez](robustness.es.md).

Use `GET /experiments?limit=50&offset=0` (límite 1–100), `GET /experiments/{id}` y `GET /experiments/{id}/report?format=json|markdown`. El informe JSON incluye `optimization_analysis` desde el esquema 1.3 y agrega `robustness_analysis` en esquema 1.4; ejecuciones anteriores siguen legibles con campos aditivos nulos. La recuperación no vuelve a ejecutar el predictor ni el análisis.

La demanda usada es una estimación fija del regresor para conexiones sintéticas de la próxima hora. El optimizador no modifica ni vuelve a entrenar modelos. La optimalidad solo se demuestra para los objetivos aditivos de déficit y las restricciones declaradas; si se alcanza el límite de pasos, se indica que la optimalidad no está probada. Consulte la [metodología de optimización](optimizacion.md) y [experimentos](experiments.md).
