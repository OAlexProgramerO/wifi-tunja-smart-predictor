# Comparación avanzada de experimentos ML

## API

Envía de dos a diez IDs únicos de experimentos guardados a `POST /ml-experiments/compare`:

```json
{
  "experiment_ids": ["id-ejecucion-a", "id-ejecucion-b"]
}
```

El endpoint lee los registros existentes `ml_experiments` y devuelve un informe JSON versionado. No modifica el historial ni los artefactos. La página Experiment Orchestration del dashboard ofrece la misma comparación y descarga JSON.

## Compatibilidad

La comparación es por pares. Dos registros son comparables solo si ambos finalizaron y tienen procedencia sintética consistente; coinciden la tarea, huella y algoritmo del dataset, versiones del proyecto y del entorno, seed, objetivo, orden de features y ventanas cronológicas de entrenamiento/evaluación. Ambos deben usar la misma partición `validation` o `test`. Los campos ausentes o inconsistentes marcan el par `INCOMPATIBLE`; el informe explica las causas y omite diferencias de métricas.

Los candidatos y sus hiperparámetros pueden diferir: son precisamente los elementos que se examinan. Las huellas de artefactos candidatos se informan como procedencia, pero no deben coincidir porque los modelos ajustados son distintos. En V8.1 no se guardaron artefactos candidatos, por lo que esas huellas no están disponibles.

## Interpretación de métricas

Nunca se comparan tareas de clasificación con regresión. En clasificación, mayor accuracy, precision, recall, F1 y ROC-AUC se interpreta como mejora. En regresión, menor MAE/RMSE y mayor R² se interpreta como mejora. Diferencias absolutas de hasta `1e-9` se consideran empate. Solo se comparan métricas finitas disponibles en ambos registros. Las ausentes o indefinidas aparecen en `unavailable_metrics`; si faltan algunas, el par es `PARTIAL` y conserva las métricas compartidas válidas.

Para 3–10 registros se generan todos los pares posibles. El estado general es `COMPARABLE` si todos los pares son comparables; `PARTIAL` si hay resultados útiles o parciales, pero no todos los pares son plenamente comparables; e `INCOMPATIBLE` si no hay ningún par comparable. No se agregan métricas distintas ni se elige un mejor modelo universal.

## Datos y límites

El motor comprueba que objetivos/información futura no aparezcan en la lista de features y que partición y filas de evaluación coincidan con el corte cronológico guardado. Usa solo metadatos y métricas persistidas: no carga datos MinTIC, no vuelve a entrenar ni recalcula métricas y no cambia predicciones. Describe ejecuciones sintéticas, no rendimiento real en Tunja. La reproducción exacta conserva los límites `TRACEABLE_NOT_SELF_CONTAINED` del informe del experimento.
