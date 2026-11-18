# Gates automáticos de evaluación de experimentos ML

Los gates V8.3 aplican umbrales configurables a métricas registradas de experimentos sintéticos ML completados. Son de solo lectura: no entrenan modelos, no cambian predicciones ni promueven candidatos.

## API

`POST /ml-experiments/gates` recibe de 2 a 10 IDs únicos y criterios para una tarea:

```json
{
  "experiment_ids": ["run-guardado-a", "run-guardado-b"],
  "task": "classification",
  "criteria": [{"metric": "f1", "operator": "gte", "threshold": 0.7}]
}
```

Las métricas permitidas para clasificación son accuracy, precision, recall, F1 y ROC-AUC. Para regresión son MAE, RMSE y R². `gte` exige un valor al menos igual al umbral; `lte` exige un valor no superior. El operador debe coincidir con la dirección habitual de la métrica (mayor para clasificación y R²; menor para MAE/RMSE). La igualdad exacta pasa.

## Resultados

Cada experimento recibe `PASSED` cuando hay evidencia válida para todos los criterios y se cumplen; `FAILED` cuando evidencia válida muestra que al menos uno no se cumple; o `INCONCLUSIVE` cuando faltan métricas o procedencia requerida, o esta es inválida. El informe incluye valores, umbrales, explicaciones, huella del dataset, partición/ventana temporal, fechas, IDs y versión de esquema.

Clasificación y regresión no pueden compartir una solicitud. Además, las ejecuciones seleccionadas deben ser compatibles según el contrato del comparador V8.2: misma huella de dataset sintético, tarea, partición, entorno, configuración y ventanas cronológicas. Si no son compatibles, cada resultado es `INCONCLUSIVE` y no se juzgan umbrales. Un gate aprobado no es una clasificación universal de modelos ni evidencia de rendimiento real.

Solo se aceptan registros de experimentos ML sintéticos. Los datos oficiales MinTIC no se usan. Estos gates no prueban significancia estadística ni validez de pronósticos reales y nunca modifican artefactos de producción ni promueven modelos automáticamente.
