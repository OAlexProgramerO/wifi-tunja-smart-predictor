# Registro reutilizable de políticas de evaluación ML

V8.4 agrega perfiles de aceptación inmutables y versionados sobre el motor de gates V8.3. Cada perfil tiene `policy_id` estable, versión entera incremental, nombre, descripción, etiquetas, fecha de creación y criterios por tarea. Puede definir criterios para clasificación, regresión o ambas.

## API

- `POST /ml-experiments/policies` crea un perfil nuevo en la versión 1.
- Enviar el mismo `policy_id` agrega la siguiente versión; las versiones existentes nunca se editan ni eliminan.
- `GET /ml-experiments/policies?limit=20&offset=0` lista la última versión de cada perfil.
- `GET /ml-experiments/policies/{policy_id}` obtiene la última versión; `?version=n` obtiene una versión histórica inmutable.
- `GET /ml-experiments/policies/{policy_id}/versions?limit=100&offset=0` lista las versiones inmutables de más reciente a más antigua.
- `POST /ml-experiments/policies/{policy_id}/versions/{version}/evaluate` recibe `experiment_id` y aplica los criterios de su tarea.

El dashboard muestra las últimas versiones, permite crear un perfil o la siguiente versión y elegir cualquier versión histórica listada para evaluar.

Los informes identifican el ID/versión exactos de la política e incluyen el perfil guardado, ID y fechas del experimento, huella del dataset sintético, partición/ventana cronológica, métricas, resultados de criterios y evidencia de compatibilidad. Las tareas no admitidas, procedencia incompatible o incompleta y métricas requeridas ausentes producen `INCONCLUSIVE`. Solo registros sintéticos completados y compatibles pueden aprobar o fallar criterios.

Las políticas se guardan en una tabla aditiva de la base SQLite ML existente. Crear una versión nunca modifica las anteriores. La evaluación reutiliza los umbrales y la compatibilidad del motor V8.3; no reentrena, promueve ni modifica modelos o registros de experimentos. Los datos oficiales MinTIC se excluyen.

## Historial de ejecuciones (V8.5)

Cada llamada al endpoint de evaluación versionado crea una ejecución inmutable, incluso con resultado `INCONCLUSIVE`. El registro guarda ID de ejecución y fecha UTC únicos, ID del experimento, política y versión exactas, estado, evidencia completa por criterio e informe con huella de datos sintéticos y procedencia de la ventana cronológica. Una versión nueva de política no modifica informes anteriores.

Use `GET /ml-experiments/policy-runs` con `limit`, `offset` y filtros opcionales `policy_id`, `policy_version`, `experiment_id`, `status`, `created_from_utc` o `created_to_utc`. Las fechas son marcas ISO inclusivas con zona UTC. Consulte una ejecución con `GET /ml-experiments/policy-runs/{run_id}` o descargue el JSON guardado desde `/report`.

Compare dos IDs con `POST /ml-experiments/policy-runs/compare`. Las ejecuciones deben usar el mismo experimento, ID de política, tarea, huella sintética, partición/ventana, versión del evaluador y conjunto de métricas/operadores; las versiones y umbrales pueden diferir. Las comparaciones compatibles muestran cambios de umbral y resultado sin clasificar modelos. Métricas ausentes producen `PARTIAL`. El contexto incompatible produce `INCOMPATIBLE`, motivos claros y omite la comparación de criterios. El dashboard ofrece filtros, detalle, descarga JSON y comparación de dos ejecuciones. La evaluación y comparación usan solo datos ML sintéticos y no modifican políticas, experimentos ni modelos.
