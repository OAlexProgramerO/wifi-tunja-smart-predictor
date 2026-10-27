# V6.2.0 — Análisis de sensibilidad y robustez

V6.2 amplía el Banco de escenarios y el optimizador entero acotado de V6.1. Una optimización guardada puede analizarse con estrés sintético de demanda de un factor a la vez (OAT) y un barrido acotado de presupuesto. El análisis reutiliza las estimaciones guardadas y el optimizador; no repite la inferencia del modelo, cambia el optimizador ni modifica artefactos.

## Método e interpretación

Para cada estimación sintética guardada \(d_i\), un caso OAT modifica una sola zona a \(d'_i=m d_i\), donde se elige un multiplicador entre 0,25 y 2,0. La estimación decimal se multiplica sin redondeo entero. Las demás zonas permanecen fijas. El reoptimizador usa el objetivo, ponderaciones, capacidades y límites por zona originales, con presupuesto entero acotado. No se combinan exhaustivamente cambios de demanda y presupuesto.

El análisis evalúa por separado la asignación de referencia original bajo cada supuesto variado y la nueva asignación optimizada bajo ese mismo supuesto. La degradación del objetivo es el puntaje reoptimizado menos el puntaje de la asignación original para ese caso; el porcentaje divide por el valor absoluto del puntaje original y queda nulo si este es cero. Así se distingue la sensibilidad de la asignación de referencia del resultado de reoptimizar.

La frontera de presupuesto solo informa presupuestos efectivamente evaluados; no interpola. El presupuesto adicional mínimo para cumplir las asignaciones mínimas declaradas es exactamente `max(0, suma de mínimos por zona - presupuesto evaluado)`. Es un cálculo de restricciones, no una recomendación.

La estabilidad por zona es la proporción de casos evaluados y analizables que tienen una solución factible del optimizador donde la asignación de esa zona coincide con la referencia guardada. Cada zona informa denominador, cantidad sin cambio, cambio absoluto medio y máximo. El orden de “zonas más sensibles” es descriptivo: máximo, luego media de cambio absoluto y luego ID de zona.

La tasa de factibilidad divide los casos evaluados factibles entre todos los casos evaluados. La integridad/procedencia del experimento padre se reporta por separado. Los casos son escenarios seleccionados deliberadamente, no una muestra aleatoria: las cantidades y tasas no son probabilidades, intervalos de confianza, casos extremos calibrados ni estimaciones de confiabilidad operativa real.

## Límites y cotas

- El padre debe ser un registro de optimización. Si la integridad es incompleta o inválida, la evaluación no se marca como verificada; los resultados numéricos conservan el estado de procedencia.
- Se admiten como máximo 12 zonas, 6 multiplicadores incluyendo la referencia 1,0 insertada, 20 presupuestos incluyendo el presupuesto padre insertado, 100 evaluaciones y 10.000 pasos del optimizador por evaluación.
- Los rangos son inclusivos. Se incluyen multiplicador 1,0 y presupuesto original como referencia si no aparecen en los rangos introducidos. Si el límite de evaluaciones omite casos o un optimizador interno alcanza su límite de pasos, el análisis queda parcial e informa el denominador evaluado y el conteo de optimizadores incompletos.
- `search_limit_reached` se conserva por ejecución del optimizador; no demuestra optimalidad.
- `optimizer_incomplete_scenario_count` cuenta los casos evaluados cuyo optimizador interno alcanzó el límite de pasos; aun si devuelve una solución factible, su optimalidad no está demostrada.
- La inviabilidad de mínimos rígidos se indica explícitamente. El presupuesto adicional se calcula a partir de esos mínimos; máximos y demanda no cambian ese cálculo.
- Datos, estimaciones, capacidades, objetivos, presupuestos y asignaciones son sintéticos o hipotéticos. Nada se ha validado con demanda real de WiFi público ni infraestructura de Tunja.

## Ejemplo de API

```json
{
  "name": "prueba acotada de estrés sintético",
  "parent_experiment_id": "<id-de-optimizacion-guardada>",
  "demand_multiplier_min": 0.8,
  "demand_multiplier_max": 1.2,
  "demand_multiplier_step": 0.2,
  "budget_min_connections": 0,
  "budget_max_connections": 1000,
  "budget_step_connections": 100,
  "max_evaluations": 100,
  "max_allocation_steps_per_evaluation": 10000
}
```

Envíe a `POST /experiments/robustness`. El resultado se guarda en el SQLite de experimentos actual y se recupera/exporta mediante las rutas existentes. El esquema de informe 1.4 agrega `robustness_analysis`; los registros anteriores siguen siendo legibles.
