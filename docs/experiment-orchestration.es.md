# Orquestación reproducible de experimentos ML

## Alcance y configuración

El servicio ejecuta un candidato permitido de clasificación o regresión con el conjunto sintético y el corte cronológico del proyecto. Evalúa en el periodo `validation` o `test`. Los objetivos se usan solo como etiquetas, nunca como características. El servicio no lee instantáneas MinTIC, no guarda artefactos ajustados, no reentrena los modelos de producción ni cambia las predicciones.

La configuración contiene `task` (`classification` o `regression`), `model_candidate` permitido, `evaluation_split` e `hyperparameters` de primitivas acotadas. Se validan los parámetros frente a las especificaciones existentes. Cada registro tiene UUID, fechas UTC, candidato, configuración, estado, huella del dataset, particiones cronológicas, métricas y huella de artefacto solo si se guardó alguno. V8.1 no persiste artefactos candidatos, así que dicha huella queda nula. Los errores guardan únicamente un tipo sanitizado y un mensaje genérico.

```text
configuración válida -> PENDING -> RUNNING -> COMPLETED
                                         \-> FAILED
```

La solicitud FastAPI se ejecuta sincrónicamente y guarda cada estado en la tabla aditiva `ml_experiments` de la base SQLite existente. No cambia registros ni esquema de experimentos de escenarios. Una interrupción puede dejar `RUNNING`; no hay trabajador de fondo ni reintento automático. Una clave de idempotencia repetida con igual configuración devuelve el registro existente; una configuración diferente se rechaza.

## API e interfaz

- `POST /ml-experiments`: valida y ejecuta; admite `X-Idempotency-Key`.
- `GET /ml-experiments?limit=20&offset=0`: historial paginado y acotado.
- `GET /ml-experiments/{experiment_id}`: detalle.
- `GET /ml-experiments/{experiment_id}/report`: informe JSON persistido.

La página Streamlit Experiment Orchestration envía ejecuciones, consulta historial/detalles y descarga informes. Es ejecución local sincrónica, no orquestación distribuida. Reproducir un resultado requiere el mismo código, dependencias y bytes del conjunto sintético. Los experimentos sobre test son para inspección, no para selección iterativa. Las métricas no validan pronósticos reales; los datos MinTIC oficiales no son entradas.
