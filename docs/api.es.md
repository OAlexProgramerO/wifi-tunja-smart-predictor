# Referencia de API — optimización de escenarios V6.1

La API principal corre en el puerto 8000; Swagger está disponible en `/docs`.

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

Use `GET /experiments?limit=50&offset=0` (límite 1–100), `GET /experiments/{id}` y `GET /experiments/{id}/report?format=json|markdown`. La exportación JSON usa esquema 1.3 e incluye `optimization_analysis`. Las ejecuciones antiguas mantienen el mismo contenido y reportan ese campo como `null`. La recuperación no vuelve a ejecutar el predictor ni la optimización.

La demanda usada es una estimación fija del regresor para conexiones sintéticas de la próxima hora. El optimizador no modifica ni vuelve a entrenar modelos. La optimalidad solo se demuestra para los objetivos aditivos de déficit y las restricciones declaradas; si se alcanza el límite de pasos, se indica que la optimalidad no está probada. Consulte la [metodología de optimización](optimizacion.md) y [experimentos](experiments.md).
