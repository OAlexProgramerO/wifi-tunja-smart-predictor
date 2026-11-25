# Release history and V4.0 readiness

## V8.10.0 — Verification history summary reports

Adds read-only deterministic JSON aggregation of local verification history by result and available export format, with matched totals, missing-format counts, date range, and result/date filters. Summary operations open the existing SQLite store read-only and expose metadata only. Counts do not indicate model quality, WiFi performance, or operational reliability. Product V8.10.0 maps to Python/Docker `0.8.10`; verification history remains separate from official MinTIC descriptive data.

## V8.10.0 — Informes resumen del historial de verificaciones

Agrega agregación JSON determinista y de solo lectura del historial local por resultado y formato disponible, con totales, formatos faltantes, rango de fechas y filtros de resultado/fecha. Los resúmenes abren SQLite en modo de solo lectura y exponen solo metadatos. Los conteos no indican calidad del modelo, rendimiento WiFi ni confiabilidad operativa. V8.10.0 corresponde a Python/Docker `0.8.10`; el historial sigue separado de los datos descriptivos oficiales MinTIC.

## V8.9.0 — Audit verification history

Adds opt-in local SQLite storage and deterministic listing, filtering, pagination, and inspection for export-verification metadata. Stored rows never contain exported audit records. Missing, corrupted, and unsupported stores return explicit errors without being reset. Product V8.9.0 maps to Python/Docker `0.8.9`; ML evidence remains synthetic-only and separate from official MinTIC descriptions.

## V8.9.0 — Historial de verificación de auditoría

Agrega almacenamiento SQLite local opcional y listado, filtros, paginación y consulta deterministas de metadatos de verificación. Las filas guardadas nunca contienen los registros de auditoría exportados. Los almacenes ausentes, dañados o incompatibles devuelven errores explícitos sin reiniciarse. V8.9.0 corresponde a Python/Docker `0.8.9`; las evidencias ML siguen siendo sintéticas y separadas de las descripciones oficiales MinTIC.

## V8.8.0 — Offline audit export verification

Adds a downloadable manifest for JSON/CSV audit exports and a standard-library CLI to verify files offline. The manifest hashes one deterministic canonical record array shared by both file formats, not the incidental JSON/CSV serialization. Verification checks schema, filters, count, and digest with `VALID`, `INVALID`, or `UNSUPPORTED` results. No database or network access is needed, and no saved audits or V8.6 digests are changed. Local digests are not authenticity or tamper-proof evidence. Product V8.8.0 maps to Python/Docker `0.8.8`; audit data remains synthetic and separate from official MinTIC records.

## V8.8.0 — Verificación offline de exportaciones de auditoría

Agrega un manifiesto descargable para exportaciones JSON/CSV de auditoría y un CLI de biblioteca estándar que verifica archivos offline. El manifiesto calcula el hash de un arreglo de registros canónico determinista compartido por ambos formatos, no de su serialización JSON/CSV incidental. La verificación comprueba esquema, filtros, cantidad y digest con resultados `VALID`, `INVALID` o `UNSUPPORTED`. No requiere base de datos ni red y no modifica auditorías guardadas ni digests V8.6. Los digests locales no prueban autenticidad ni invulnerabilidad a cambios. V8.8.0 corresponde a Python/Docker `0.8.8`; la auditoría sigue sintética y separada de MinTIC oficial.

## V8.7.0 — Audit evidence export

Adds filtered JSON/CSV downloads for all matching local policy audit events, with export metadata, deterministic record order, original digests, and verification states. Explicit allowlisting excludes unknown event properties, and dashboard exports are independent of history pagination. Exports are read-only and preserve the V8.6 consistency-only digest limitation. Product V8.7.0 maps to Python/Docker `0.8.7`; synthetic audit events remain separate from official MinTIC data.

## V8.7.0 — Exportación de evidencias de auditoría

Agrega descargas JSON/CSV filtradas de todos los eventos locales de auditoría de políticas coincidentes, con metadatos, orden determinista, digests originales y estados de verificación. La lista explícita de campos excluye propiedades desconocidas; las descargas del dashboard no dependen de la paginación del historial. Las exportaciones son de solo lectura y conservan el límite de consistencia local del digest V8.6. V8.7.0 corresponde a Python/Docker `0.8.7`; los eventos sintéticos permanecen separados de los datos oficiales MinTIC.

## V8.6.0 — Evaluation audit trail and integrity verification

Records privacy-safe local audit events atomically with new policy evaluation runs. Deterministic canonical JSON and SHA-256 digests support paginated/filterable history and verification of event, index, and linked run/report consistency. Verification results distinguish `VERIFIED`, `INVALID`, and `UNVERIFIABLE`; older runs are left unchanged. The digest is not a signature or tamper-proof mechanism, and a privileged database operator could rewrite both records and digests. Audit data covers synthetic ML evaluations only, separate from official MinTIC records. Product V8.6.0 maps to Python/Docker `0.8.6`.

## V8.6.0 — Auditoría de evaluaciones y verificación de integridad

Registra eventos locales privados de auditoría en la misma transacción que nuevas ejecuciones de políticas. JSON canónico y digest SHA-256 deterministas permiten consultar historial paginado/filtrable y verificar consistencia entre evento, índice y ejecución/informe enlazado. Los estados distinguen `VERIFIED`, `INVALID` y `UNVERIFIABLE`; las ejecuciones anteriores permanecen intactas. El digest no es una firma ni un mecanismo invulnerable a modificaciones; un operador privilegiado podría reescribir ambos registros y digests. La auditoría cubre evaluaciones ML sintéticas y se mantiene separada de MinTIC oficial. V8.6.0 corresponde a Python/Docker `0.8.6`.

## V8.5.0 — Evaluation policy run history — released 2026-11-20

Persists every policy evaluation as an immutable report/run ID with timestamp, experiment and exact policy version, result, criterion evidence, and synthetic provenance. Adds filtered/paginated history, detail and JSON download routes, and comparison limited to compatible experiment/policy/dataset/evaluation/evaluator contexts. Missing criterion evidence is partial; incompatible records include reasons without criterion comparisons. Dashboard history supports filters, detail inspection, report download, and run comparison. No models or experiment records are modified/promoted; MinTIC is excluded. Product V8.5.0 maps to Python/Docker `0.8.5`. Annotated tag [`v8.5.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.5.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.5.0) target `8aa5b88c19797acbf93dc95d1086b9b439d7f944`. Pytest, Ruff, Black, Compose configuration, and `git diff --check` passed locally.

## V8.5.0 — Historial de ejecuciones de políticas de evaluación — publicada 2026-11-20

Persiste cada evaluación de política como ejecución/informe inmutable con ID, fecha, experimento, versión exacta de política, resultado, evidencia por criterio y procedencia sintética. Agrega historial filtrable/paginado, rutas de detalle/descarga JSON y comparación limitada a contextos compatibles de experimento, política, dataset, evaluación y evaluador. La evidencia faltante produce resultado parcial; los registros incompatibles explican motivos y no comparan criterios. El dashboard permite filtrar historial, consultar detalles, descargar informes y comparar ejecuciones. No modifica/promueve modelos o registros de experimentos; excluye MinTIC. V8.5.0 corresponde a Python/Docker `0.8.5`. La etiqueta anotada [`v8.5.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.5.0) y el [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.5.0) apuntan a `8aa5b88c19797acbf93dc95d1086b9b439d7f944`. Pasaron localmente pytest, Ruff, Black, la configuración Compose y `git diff --check`.

## V8.4.0 — Reusable evaluation policy registry — released 2026-11-19

Adds immutable, named classification/regression acceptance profiles with metadata and numbered versions in the existing ML SQLite store. Paginated APIs list profiles and versions, retrieve exact versions, and evaluate one completed compatible synthetic experiment through the existing V8.3 gate engine. Reports include the exact policy ID/version/criteria and experiment/dataset/chronological provenance. The dashboard creates profiles/versions, browses historical versions and exports evaluation reports. Unsupported task types or missing/incompatible evidence are inconclusive. No model or experiment record is changed or promoted; official MinTIC data is excluded. Product V8.4.0 maps to Python and Docker `0.8.4`. Annotated tag [`v8.4.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.4.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.4.0) target `9c970076c09422c5632444be4ccf646b22f7d5f6`. Pytest, Ruff, Black, Compose configuration, and `git diff --check` passed locally.

## V8.4.0 — Registro reutilizable de políticas de evaluación — publicada 2026-11-19

Agrega perfiles inmutables y nombrados de aceptación para clasificación/regresión, con metadatos y versiones numeradas en la base ML SQLite existente. La API paginada lista perfiles/versiones, recupera versiones exactas y evalúa un experimento sintético finalizado compatible mediante el motor V8.3. Los informes incluyen ID/versión/criterios exactos y procedencia del experimento, dataset y ventana cronológica. El dashboard crea perfiles/versiones, consulta versiones históricas y exporta informes. Tareas no admitidas o evidencia ausente/incompatible quedan inconclusas. No modifica ni promueve modelos o registros; los datos oficiales MinTIC se excluyen. V8.4.0 corresponde a Python y Docker `0.8.4`. La etiqueta anotada [`v8.4.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.4.0) y el [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.4.0) apuntan a `9c970076c09422c5632444be4ccf646b22f7d5f6`. Pasaron pytest, Ruff, Black, la configuración Compose y `git diff --check`.

## V8.3.0 — Automated experiment evaluation gates — released 2026-11-18

Adds read-only, task-specific acceptance thresholds for compatible persisted synthetic ML experiments. Each criterion reports value, threshold, explanation, and `PASSED`, `FAILED`, or `INCONCLUSIVE`; reports include synthetic dataset fingerprints and chronological evaluation provenance. API and existing Experiment Orchestration dashboard support configuration and JSON export. Incompatible records are inconclusive; no model is promoted and no artifact is modified. Product V8.3.0 maps to Python and Docker `0.8.3`. Annotated tag [`v8.3.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.3.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.3.0) target `1f543cd82cd2343bf1591ff72ee2956b124b04c8`. Pytest, Ruff, Black, Compose configuration, and `git diff --check` passed locally. All ML records use synthetic data; official MinTIC data is excluded.

## V8.3.0 — Gates automáticos de evaluación de experimentos — publicada 2026-11-18

Agrega umbrales de aceptación por tarea, de solo lectura, para experimentos ML sintéticos guardados compatibles. Cada criterio informa valor, umbral, explicación y estado `PASSED`, `FAILED` o `INCONCLUSIVE`; los reportes incluyen huellas del dataset sintético y procedencia de evaluación cronológica. La API y el dashboard existente permiten configurar y exportar JSON. Los registros incompatibles quedan inconclusos; ningún modelo se promueve ni se modifica artefacto. V8.3.0 corresponde a Python y Docker `0.8.3`. La etiqueta anotada [`v8.3.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.3.0) y el [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.3.0) apuntan a `1f543cd82cd2343bf1591ff72ee2956b124b04c8`. Pasaron localmente pytest, Ruff, Black, la configuración Compose y `git diff --check`. Los registros ML usan solo datos sintéticos; se excluyen datos oficiales MinTIC.

## V8.2.0 — Advanced experiment comparison — released 2026-11-17

Adds read-only pairwise comparisons of completed synthetic classification or regression runs. Compatibility checks cover dataset fingerprint and algorithm, task/target/features/seed, project and runtime versions, and chronological training/evaluation windows; incompatible pairs include reasons and have no metric deltas. Compatible comparisons show task-specific metric directions, improvements/regressions/ties, unavailable metrics, model configuration, experiment timestamps, dataset provenance, and artifact fingerprint status. The endpoint and dashboard JSON download never modify run history or production artifacts and do not select a universal best model. Product V8.2.0 maps to Python and Docker version `0.8.2`. Annotated tag [`v8.2.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.2.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.2.0) target `ec2a1e830888a2927d2dcbbd328763f7bd8f2fcf`. Pytest, Ruff, Black, Compose configuration, and `git diff --check` passed locally. All ML data is synthetic; official MinTIC data remains separate.

## V8.2.0 — Comparación avanzada de experimentos — publicada 2026-11-17

Agrega comparaciones de solo lectura por pares de ejecuciones sintéticas finalizadas de clasificación o regresión. Valida huella/algoritmo del dataset, tarea/objetivo/features/seed, versiones del proyecto y entorno y ventanas cronológicas; los pares incompatibles muestran causas y omiten diferencias. Las comparaciones compatibles indican dirección de métricas por tarea, mejoras/regresiones/empates, métricas no disponibles, configuración del modelo, fechas, procedencia del dataset y estado de huella de artefactos. La API y descarga JSON del dashboard no modifican historial ni artefactos de producción ni seleccionan un mejor modelo universal. V8.2.0 corresponde a las versiones Python y Docker `0.8.2`. La etiqueta anotada [`v8.2.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.2.0) y el [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.2.0) apuntan a `ec2a1e830888a2927d2dcbbd328763f7bd8f2fcf`. Pasaron localmente pytest, Ruff, Black, la configuración Compose y `git diff --check`. Los datos ML son sintéticos; los datos oficiales MinTIC permanecen separados.

## V8.1.0 — Reproducible ML experiment orchestration — released 2026-11-16

Adds synchronous, allowlisted classification/regression runs on synthetic data with validated bounded configuration, UTC lifecycle timestamps, chronological split details, dataset fingerprint, metrics, sanitized failure state, and idempotent persistence in an additive SQLite table. The API supports bounded list, detail, and JSON report retrieval; the dashboard adds an orchestration page. Candidate artifacts are not persisted and production predictions are unchanged. Reports are marked `TRACEABLE_NOT_SELF_CONTAINED`; exact replay is not verified and requires the same code, recorded libraries, and dataset bytes. MinTIC data remains excluded. Product V8.1.0 maps to Python package and Docker image `0.8.1`. Annotated tag [`v8.1.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.1.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.1.0) target `579d14da34cf64511385d4410b912c066bedbb09`. Pytest, Ruff, Black, Compose configuration, and `git diff --check` passed locally.

## V8.1.0 — Orquestación reproducible de experimentos ML — publicada 2026-11-16

Agrega ejecuciones sincrónicas permitidas de clasificación/regresión sobre datos sintéticos, con configuración acotada y validada, fechas UTC de ciclo de vida, partición cronológica, huella del dataset, métricas, errores sanitizados y persistencia idempotente en una tabla SQLite aditiva. La API permite listar con límites, consultar detalles y obtener informes JSON; el dashboard agrega una página de orquestación. Los artefactos candidatos no se guardan y las predicciones de producción no cambian. Los informes usan `TRACEABLE_NOT_SELF_CONTAINED`: no se verifica reproducción exacta y se requieren el mismo código, bibliotecas registradas y bytes del dataset. Se excluyen los datos MinTIC. V8.1.0 corresponde al paquete Python y la imagen Docker `0.8.1`. La etiqueta anotada [`v8.1.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.1.0) y el [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.1.0) apuntan a `579d14da34cf64511385d4410b912c066bedbb09`. Pasaron localmente pytest, Ruff, Black, la configuración Compose y `git diff --check`.

## V8.0.0 — Synthetic Data Drift Monitoring — released 2026-11-15

Adds reproducible feature-level distribution reports comparing synthetic training features with a separately selected validation or test split. The read-only API and dashboard expose numerical/categorical summaries, missingness, unseen categories, sample adequacy, thresholds, and feature-only fingerprints; an explicit CLI command stores versioned JSON reports. Targets, future information, and official MinTIC records are excluded. Drift is a statistical description of these synthetic partitions, not evidence of real-world model degradation or changes in public WiFi demand. Product V8.0.0 maps to Python package and Docker image `0.8.0`. Annotated tag [`v8.0.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.0.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.0.0) target `9de3becaca53546baa7e3438f3527832107fd923`. `pytest -q`, `ruff check .`, `black --check .`, `docker compose config --quiet`, and `git diff --check` passed locally.

## V8.0.0 — Monitoreo de deriva de datos sintéticos — publicado 2026-11-15

Agrega informes reproducibles por característica que comparan variables de entrenamiento sintético con una partición validation o test seleccionada por separado. La API de solo lectura y el dashboard muestran resúmenes numéricos/categóricos, faltantes, categorías nuevas, suficiencia de muestras, umbrales y huellas calculadas solo con características; un comando CLI explícito guarda informes JSON versionados. Se excluyen objetivos, información futura y registros MinTIC oficiales. La deriva describe estadísticamente estas particiones sintéticas; no demuestra degradación real del modelo ni cambios en la demanda de WiFi público. V8.0.0 corresponde al paquete Python y la imagen Docker `0.8.0`. La etiqueta anotada [`v8.0.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v8.0.0) y el [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v8.0.0) apuntan a `9de3becaca53546baa7e3438f3527832107fd923`. Pasaron localmente `pytest -q`, `ruff check .`, `black --check .`, `docker compose config --quiet` y `git diff --check`.


## V7.9.0 — Model run comparison — released 2026-11-14

Adds read-only comparisons of persisted model candidates within classifier and regressor tasks. Deltas are omitted for incompatible fingerprints or evaluation windows. The latest saved tables are used; this is not an archive of historical training runs. Current persisted records report `PARTIAL` where run IDs/evaluation timestamps are absent. Product V7.9.0 maps to Python package and Docker image `0.7.9`. Annotated tag [`v7.9.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.9.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.9.0) target `35139bd10d5cbfb30700bd71be53ab9b7113a057`. Pytest, Ruff, Black, and `git diff --check` passed.

## V7.8.0 — ML artifact inventory — released 2026-11-13

Adds read-only classifier/regressor artifact availability, metadata consistency, and SHA-256 fingerprint checks, with API diagnostics and an About-page summary. It never retrains or modifies model files. Product V7.8.0 maps to Python package and Docker image `0.7.8`. Annotated tag [`v7.8.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.8.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.8.0) target `ab95a82cdf4be121b9b8745c2ed3154cae7aa286`.

## V7.7.0 — Operational observability — released 2026-11-12

Adds bounded process-local API request, response-status, error, and duration aggregates, plus read-only diagnostics and a compact dashboard summary. The main API checks the assistant metrics/readiness route and distinguishes unavailable checks from verified not-ready/failure states. No request content or sensitive headers are retained. Metrics reset on process restart and are not durable monitoring. Synthetic ML inputs and official MinTIC descriptive analytics remain separate. Product V7.7.0 maps to Python package and Docker image `0.7.7`. Annotated tag [`v7.7.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.7.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.7.0) target `e57c7d321f3302b7228228425d52b3e65d3719dd`. Pytest, Ruff, Black, Compose configuration, and `git diff --check` passed locally.

## V7.6.0 — Architecture and system diagnostics — released 2026-11-11

Adds read-only `GET /architecture/diagnostics` and an About-page component status view. Only reliable checks are marked verified; unavailable checks are distinct from verified errors. Docker daemon state and sensitive configuration are not exposed. The system architecture guide describes service responsibilities, flows, Compose relationships, and the separation of synthetic ML resources from official MinTIC records. Product V7.6.0 maps to Python package and Docker image `0.7.6`. Annotated tag [`v7.6.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.6.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.6.0) target `2749659ff79bf3a063314c7581660f9ce361c77b`. Pytest, Ruff, Black, and Compose configuration passed; Docker image build/runtime was not executed.

## V7.5.0 — Provider concentration analysis — released 2026-11-10

Adds selected-period provider shares, leading provider, provider count, and HHI for MinTIC fixed-internet access records. Exact duplicate source rows are excluded; selected-period incomplete or invalid fields and zero denominators result in unavailable/undefined metrics rather than imputation. This describes municipal subscription statistics only and does not measure WiFi traffic or network performance. Product V7.5.0 maps to Python package and Docker image `0.7.5`. Annotated tag [`v7.5.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.5.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.5.0) target `e85b2471afe6b88ced6345161c2f4c28e4a4441c`. Local pytest, Ruff, Black, Compose configuration, and `git diff --check` passed. See the [V7.5.0 validation record](releases/v7.5.0.md).

## V7.4.0 — Official Data Quality & Coverage — released 2026-11-09

Adds a provenance-aware MinTIC snapshot quality report in the dashboard and `GET /connectivity/tunja/quality`. Reports source/accepted/rejected counts, missing fields, exact duplicates, observed periods, incomplete-period indicator availability, and validation status. Missing information remains unavailable; official records are not edited and remain separate from synthetic ML. Product V7.4.0 maps to Python package and Docker image `0.7.4`. Annotated tag [`v7.4.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.4.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.4.0) target `46cfed52a54d6c6248531069b8e2d1dea4e77c25`. Local pytest, Ruff, Black, and `git diff --check` passed. See the [V7.4.0 validation record](releases/v7.4.0.md).

## V7.3.0 — Official Connectivity Period Comparison — released 2026-11-08

Adds dashboard period selectors and a read-only API comparison for available official MinTIC Tunja periods. Reports access-count aggregates by provider, technology, and bandwidth category with absolute and valid percentage differences; missing periods/categories are not interpolated or zero-filled, and a zero baseline has no percentage result. Municipal subscriptions do not measure WiFi traffic, congestion, or saturation; ML remains synthetic. Product V7.3.0 maps to Python package and Docker image `0.7.3`. Annotated tag [`v7.3.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.3.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.3.0) target `aa4fb65eb7072c8b4396e034966c646555eedb18`. Local pytest, Ruff, Black, and `git diff --check` passed. Docker build and live runtime were not run. See the [V7.3.0 validation record](releases/v7.3.0.md).

## V7.2.0 — Official Data Refresh History — released 2026-11-07

Adds append-only refresh attempt tracking for the official MinTIC snapshot, read-only latest-status and paginated-history API routes, and a dashboard status/history view. Failed attempts preserve the last valid local snapshot; cached/stale data are distinguished from refresh success and never called live. Package and Docker image version `0.7.2`. Annotated tag [`v7.2.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.2.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.2.0) target `785fa9d01fd4dd16b0b5356bedca79afe65e22a2`. Local pytest, Ruff, Black, compileall, Compose configuration, and `git diff --check` passed. See the [V7.2.0 validation record](releases/v7.2.0.md).

## V7.1.0 — Official Connectivity Data Exports — released 2026-11-06

Adds CSV downloads of accepted MinTIC records and JSON downloads separating raw source rows, normalized rows, analysis, and provenance. Empty/incomplete records and stale cache are identified safely; no cached data are called live. Package and Docker image version `0.7.1`. Annotated tag [`v7.1.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.1.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.1.0) point to `0ff5e340c30076ac472f74c8fba72eb2f40dfdef`. Local pytest, Ruff, Black, compileall, Compose configuration, and `git diff --check` passed. Docker image build/runtime were not run in this release workflow. See the [V7.1.0 validation record](releases/v7.1.0.md).

## V7.0.0 — Real Connectivity Intelligence — released 2026-11-05

Adds a separately labeled, optional MinTIC fixed-internet connector for official municipality-level Tunja access records, raw/normalized SQLite snapshots with provenance, offline reuse, read-only API endpoints, a dedicated dashboard, and source-grounded deterministic assistant answers. Existing ML remains synthetic; no saturation proxy or neighborhood observations are claimed. CRC Postdata QoE and ONTIC resources remain references because a stable structured Tunja feed/layer was not verified. Product V7.0.0 maps to Python package and Docker image `0.7.0`. Annotated tag [`v7.0.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v7.0.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v7.0.0) target `33c76266d08077566f4043de393afb655de66c5b`. Local tests and static checks passed; Docker build and runtime were not verified because Docker Engine was unavailable. See the [V7.0.0 validation record](releases/v7.0.0.md).

## V6.5.0 — Model metrics guide — released 2026-10-30

Adds concise Model Performance hover explanations for Accuracy, Precision, Recall, F1-score, MAE, RMSE, and R². Evaluation calculations and synthetic-data disclaimer are unchanged. Product V6.5.0 uses Python package and Docker image `0.6.5`. Annotated Git tag [`v6.5.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.5.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.5.0) target `4b7a243b005243531ce9e8d3e84830f1d23056b3`. Focused and full pytest suites, Ruff, Black, and `git diff --check` passed locally; CI was not run for this commit. Metrics remain synthetic. See the [validation record](releases/v6.5.0.md).

## V6.4.0 — Clearer model evaluation summary — released 2026-10-29

Improves the existing Model Performance presentation of model version and headline classification metrics. Evaluation methodology, report contract, model artifacts, and predictions are unchanged. Product V6.4.0 uses Python package and Docker image `0.6.4`. Annotated Git tag [`v6.4.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.4.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.4.0) target verified commit `8841996993aec18405f19e143796175cafa51540`; both CI jobs passed in [workflow 40967789602](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40967789602). Metrics remain synthetic. See [model diagnostics](model-diagnostics.md) and [validation record](releases/v6.4.0.md).

## V6.3.0 — Temporal evaluation and synthetic drift lab — released 2026-10-28

Adds bounded expanding chronological evaluation, synthetic feature-distribution diagnostics, report provenance, a typed API endpoint, and Model Performance dashboard workflow. Product V6.3.0 uses Python package and Docker image `0.6.3`. Annotated Git tag [`v6.3.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.3.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.3.0) target verified commit `e6250f81b0bbe64c404cb9401b5a3a340ed58ff9`; both CI jobs passed in [workflow 40781788884](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40781788884). All evaluations are synthetic and do not validate real Tunja WiFi demand. See [temporal evaluation](temporal-evaluation.md) and [validation record](releases/v6.3.0.md).

## V6.2.0 — Sensitivity and robustness — released 2026-10-27

Adds bounded synthetic demand sensitivity and budget sweeps for saved optimization runs. Product V6.2.0 uses Python package and Docker image `0.6.2`. Annotated Git tag [`v6.2.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.2.0) and the [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.2.0) target verified commit `4e218cba38e80ef22e6678d6b90d43ae10c313a9`; both jobs passed in [workflow 40597676230](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40597676230). Selected scenarios do not represent real WiFi demand or real-world operational reliability in Tunja. See the [V6.2.0 validation record](releases/v6.2.0.md).

## V6.1.0 — Intelligent scenario optimization — released 2026-10-26

Adds deterministic bounded integer allocation optimization to the existing synthetic scenario workbench and API, persisting through the existing SQLite experiment records and adding optimization details to report schema 1.3. Product V6.1.0 maps to Python package/Docker image `0.6.1`. Annotated Git tag [`v6.1.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.1.0) points to verified release commit `29b0814e370ac3c57467ace343acd66de9323e8d`; the [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.1.0) is published. Local checks and both CI jobs passed in [workflow 40418511852](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40418511852); local Docker Engine was unavailable, with container validation completed by CI. All demand and allocations are synthetic and do not validate real WiFi demand in Tunja. See [optimization guide](optimization.md), [Spanish guide](optimizacion.md), and the [V6.1 validation record](releases/v6.1.0.md).

# V6.0.0 — Geospatial intelligence and what-if simulation — released 2026-10-25

Adds historical synthetic geospatial analysis, a deterministic multi-zone what-if workflow, persisted comparison records, and scenario-aware reports. Product version `6.0.0` maps to Python package and Docker image `0.6.0`. Annotated tag [`v6.0.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.0.0) points to release commit `39c10c450d205a3cd9b70f9119d99db291c51cd1`; the [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.0.0) is published. Local tests and quality checks passed, and both jobs passed in [workflow 40240295823](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40240295823). The Docker daemon was unavailable locally; remote CI performed container validation. Synthetic observations and outputs do not validate real public WiFi demand or service in Tunja. See [the V6.0.0 validation record](releases/v6.0.0.md).

## V5.3.0 — Experiment observability and auditability — released 2026-10-24

Adds derived experiment integrity status, reason-coded comparison failures, safe history warnings, report validation, and bilingual dashboard status. Product V5.3.0 uses package and Docker image `0.5.3`. Annotated tag [`v5.3.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v5.3.0) points to release commit `20fb2c6e140ce774843abbb9449c4d0889d39227`; the [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v5.3.0) is published. Local checks and both remote CI jobs passed in [workflow 40138776914](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40138776914). All experiment values remain synthetic. See [the V5.3.0 validation record](releases/v5.3.0.md).

## V5.2.0 — Engineering upgrade — released 2026-10-23

Adds explicit comparison interpretation for compatible experiments with or without shared synthetic zone IDs, plus bounded page navigation in Experiment History. Product V5.2.0 uses Python package and Docker image `0.5.2`. Annotated tag [`v5.2.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v5.2.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v5.2.0) point to release commit `64c4d9369dbe8efbf69e808d2cc82616f60b8dfd`. Local checks and remote CI passed; Docker runtime, HTTP smoke, and persistence passed in [workflow 40039016215](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40039016215). All data and metrics remain synthetic. See the [V5.2.0 validation record](releases/v5.2.0.md).

## V5.1.0 — Stabilization Patch — released 2026-10-22

Fixes experiment database connection cleanup, rejects malformed persisted comparison provenance/metrics safely in the API and assistant, makes API/assistant default experiment storage independent of the process working directory, and refreshes stale evaluation reports against an integrity-validated existing model bundle without retraining. It also fixes Docker experiment-volume permissions and classifier/regressor fingerprint matching. Product V5.1.0 uses Python package and Docker image version `0.5.1`. The annotated tag [`v5.1.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v5.1.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v5.1.0) were published separately. See the [V5.1.0 validation record](releases/v5.1.0.md). All observations and predictions remain synthetic.

## V5.0.0 — implementation candidate; not released

Adds deterministic synthetic capacity planning, multi-zone scenario execution through the existing model service, SQLite experiment persistence, API retrieval and reporting, and dashboard experiment history. The package and image candidate version is `0.5.0`. Docker runtime, security audit, and remote CI require fresh verification. No V5.0.0 tag or GitHub Release is claimed. All analysis remains synthetic. See the [V5.0.0 validation record](releases/v5.0.0.md), [capacity-planning guide](capacity-planning.md), and [experiment guide](experiments.md).

## V4.4.0 — implementation candidate; not released

Adds a provenance-checked diagnostics report with chronological expanding-window fits, an additive API endpoint, Model Performance views, and deterministic assistant metric answers. Python package and Docker image candidate versions are `0.4.4`. Docker/runtime and remote CI gates must be verified before a stable tag or GitHub Release. Metrics remain limited to synthetic data. See [V4.4.0 validation record](releases/v4.4.0.md) and [diagnostics documentation](model-diagnostics.md).

## V3.8 to V3.9: changes confirmed by Git history

`v3.8.0` points to `e935b33`; `v3.9.0` points to `16128ff` and is present on both the local repository and `origin`. The V3.9 range contains eight commits and changes 31 files (3,296 insertions, 166 deletions). The classifier, regressor, synthetic dataset methodology, scenario logic, and dashboard product sections predate V3.9; they were not introduced by the Docker work.

| V3.9 change | Evidence in the repository | Verification and remaining limit |
| --- | --- | --- |
| Compose application image and services | `Dockerfile`, `compose.yaml`; commits `af035aa` and `e23ae20` | `docker compose config --quiet` passed in this V4.0 audit. Building/running needs an active Docker engine. |
| Synthetic runtime initialization and persistent data/model/report volumes | `scripts/initialize_runtime.py`, `compose.yaml`; initializer checks required file presence, runs the existing generator/preparation/training scripts once, and refuses partial resources | Unit tests cover complete, partial, and failed initialization. This is file-presence idempotency, not validation of artifact provenance or content hashes. |
| Health and readiness coordination | At V3.9, `api/routes/prediction.py`, `assistant_api/main.py`, `src/wifi_tunja_smart_predictor/deployment.py` checked runtime file presence. | V4.0 extends readiness to deserialize both models and validate complete, matching feature/version metadata; tests cover corrupt model files and version mismatches. |
| Compose-aware dashboard API clients and HTTP smoke tool | `src/wifi_tunja_smart_predictor/api_client.py`, `scripts/smoke_test.py`, dashboard route changes | Unit tests cover clients and smoke criteria; the live container-network smoke test needs Docker. |
| Dependency source/lock and CI | `pyproject.toml`, requirements wrappers, hash-locked `requirements.lock`, `.github/workflows/ci.yml`; commits `a565bbf`, `01f70a3` | CI defines a build/start/smoke/restart persistence job. The workflow file alone does not prove a successful remote run. |
| Deployment, troubleshooting and validation documentation | READMEs, `docs/deployment.md`, architecture/API docs and the V3.9 acceptance plan | Documentation accurately marks Docker runtime checks pending for this audit session. |

The commit sequence (`0d3ca1f` through `16128ff`) shows that V3.9 progressed through planning, Compose implementation, documentation, dependency standardization, smoke/CI checks, and a final limitations update. The V3.9 tag does not itself prove every checklist item passed.

## Why GitHub showed V3.8

The source of that label was found in both default-branch README files: they explicitly said “Current release: 0.3.8 (V3.8)” / “Versión actual: 0.3.8 (V3.8)” despite the later V3.9 tag. Those lines are corrected. A separate check of GitHub's public Releases page on 2026-10-13 showed “There aren’t any releases here.” Therefore the observed state is:

| Version surface | Observed value | Meaning |
| --- | --- | --- |
| Local and remote Git tag | `v3.9.0` → commit `16128ff` | Tag exists and is pushed to `origin`. |
| GitHub Release | None listed | No release page/package was published through GitHub Releases. |
| Python package | `0.3.8` in `pyproject.toml` and package metadata | Installed application version, not a Git tag. |
| Docker image | `wifi-tunja-smart-predictor:0.3.8` in Compose | Local image tag; it is not evidence of a registry push. |
| README landing-page label (before this update) | V3.8 / 0.3.8 | Stale prose, corrected to distinguish project tag from package/image and GitHub Release. |
| Automated publishing/CI evidence | `.github/workflows/ci.yml` defines test and container-smoke jobs, but not a release/image-publish workflow. GitHub Actions displayed “There are no workflow runs yet” when checked. | The checked-in workflow is configured, but no public run result was available to confirm it passed. There is no automatic GitHub Release or registry publication configured. |

The GitHub webpage evidence is the public [Releases page](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases). `gh` is not installed in this environment, so the page was checked directly; remote Git references were separately checked with `git ls-remote --tags origin`. No existing tag was moved or removed.

## V4.0 audit and chosen scope

The review found a user-facing traceability defect: `POST /predict` returned the installed package version as `model_version`, while its event log used the persisted classifier's training metadata and scenario predictions used that metadata. Persistent model volumes deliberately survive image rebuilds, so those versions can differ. The response now reports only the version recorded in valid classifier metadata; missing, malformed, incomplete, or inconsistent metadata makes prediction unavailable with a sanitized `503` rather than substituting the application version.

The readiness endpoints also previously treated the presence of files as proof that the application could use its models. V4.0 now deserializes both artifacts, checks their prediction methods, and validates the feature contract and consistency between combined, classifier, and regressor metadata. `/health` remains a liveness check; `/ready` checks application/model readiness; `/model-info` reports classifier loadability and artifact version separately from the running package version.

The ignored local artifacts provide direct evidence of the mismatch: model metadata records project version `0.3.0`, classifier `random_forest`, regressor `random_forest_regressor`, dataset `synthetic_seed_42_v1`, and training time `2026-09-30T15:53:25-05:00`; the installed package reports `0.3.8`. A local FastAPI `TestClient` prediction against that persisted classifier returned HTTP 200 and `model_version: 0.3.0`, while `/health` and `/model-info` returned application version `0.3.8`.

The checked-in local metric artifacts describe a 7,574-row synthetic temporal test set from 2025-10-01 through 2025-12-31: classification F1 for `HIGH` 0.9049 and ROC-AUC 0.9804; regression MAE 27.906, RMSE 52.860, and R² 0.8358. The nominal 90% interval's empirical synthetic holdout coverage is 0.8957. These are stored results from the artifact generation dated 2026-09-30, not metrics regenerated or independently reevaluated for V4.0 and not evidence of real-world WiFi performance.

| Functionality | Confirmed issue | Chosen improvement | Acceptance check |
| --- | --- | --- | --- |
| Standard prediction response | `model_version` could describe a newer app image than the persisted model; absent/bad metadata could be hidden by a version fallback | Return only the recorded artifact `project_version`; reject unavailable or inconsistent metadata | Tests cover a different model version plus missing, malformed, and incomplete metadata. |
| Readiness and model status | File presence could mark an unreadable or metadata-inconsistent model bundle ready | Deserialize both artifacts and validate metadata before readiness succeeds; preserve `/health` as process liveness | Tests cover corrupt artifacts, mismatched versions, and readiness failure while health remains responsive. |
| Release/version documentation | README called V3.8 current even though `v3.9.0` exists remotely; GitHub Releases is empty | Correct both READMEs, changelog, deployment note; add evidence matrix and release plan | Verify local/remote refs, page state, package/image declarations, and docs agree. |
| Docker onboarding | Deployment reference assumes prior familiarity with volumes, health/readiness, Desktop/WSL2, and resource inspection | Add a junior developer guide using this repository's actual service names and safe commands | Documentation names working Compose commands and warns that `down -v` deletes persistent demo data. |

V4.0 remains an unreleased working label. The package stays `0.3.8` and image stays `0.3.8` until the required acceptance run is complete. If accepted, keep Python's pre-1.0 SemVer package version as `0.4.0`, use project Git tag `v4.0.0`, and build image `wifi-tunja-smart-predictor:0.4.0`; publish a GitHub Release only after explicit authorization. Never overwrite `v3.9.0`.

## Pending release acceptance

During this audit, `python -m pytest -q` passed all 179 tests; Ruff, Black, compileall, `docker compose config --quiet`, and `git diff --check` also passed. The existing `scripts/smoke_test.py` was run but failed because the main API, assistant API, and dashboard were not running locally. The Docker CLI and Compose are installed, but the Docker Desktop Linux engine pipe is unavailable, so image build, isolated initialization, service health, container-network smoke, and persistence after recreation remain unverified. Dependency auditing with `uvx --from pip-audit pip-audit -r requirements.lock` could not complete because the Linux-targeted lock attempts to build `uvloop`, which does not support this Windows host. No vulnerability-free result is claimed. GitHub Actions has no observed run for this checkout.

Before assigning a V4.0 package version or release tag, start Docker Desktop and run the documented build/start/smoke/recreate checks from [deployment](deployment.md), without deleting any existing volume. Run a dependency scan from a compatible Linux environment and inspect an actual CI run. No release, image push, tag, or publication was performed as part of this work.

## V4.1 implementation status

V4.1.0 work adds bounded reliability fixes without changing the classifier/regressor algorithms, training split, synthetic data, model metrics, or dashboard sections:

- The classifier cache now includes a file signature in its cache key, so a replaced model artifact is loaded again instead of continuing to serve an estimator cached only by path.
- Model readiness and `/model-info` require the classifier's supported `LOW`/`HIGH` classes. Regression status also validates its feature metadata and consistency with the combined model record.
- The one-time runtime initializer validates an already complete model bundle. If existing artifacts fail validation, it exits unsuccessfully and leaves every persistent resource untouched.
- Unexpected prediction errors retain bounded stack-frame fields in structured diagnostics without exposing exception details in the HTTP response.

Regression tests cover artifact replacement, invalid classifier labels, missing/malformed/incomplete regression metadata, corrupt regression/classifier artifacts, inconsistent nested regression contracts, safe initialization refusal, model-info consistency, and sanitized unexpected errors. The ML pipeline was reviewed: `temporal_split` sorts and partitions by timestamp, model pipelines fit on training rows, and test rows are used for reporting rather than model selection. This review does not establish real-world performance; all data and metrics remain synthetic.

### V4.1 verification record

See the dedicated [V4.1.0 implementation and validation record](releases/v4.1.0.md) for its exact commands and results. The annotated Git tag `v4.1.0` now points to commit `edc10f03b898217d7c0108115d0d111e53962d86`; the public GitHub Releases API still lists no published releases. The V4.1 package/image baseline was `0.3.8`; Docker runtime checks and an actual CI run were not independently confirmed for that cycle. The tag does not change those historical verification results.

## V4.2.0 implementation candidate

V4.2.0 adds exact-byte model fingerprints and prediction trace metadata, a shared training-run identity for classifier/regressor bundles, stricter dataset and metric validation, clearer classification/regression presentation, and expanded CI checks. The project label is `V4.2.0`; the Python package and Compose image candidate use `0.4.2`. Local quality checks pass, while live Docker validation and remote Actions results remain pending. No V4.2.0 tag or GitHub Release exists. See the [V4.2.0 validation record](releases/v4.2.0.md) for exact evidence and release gates. All model metrics and data remain synthetic.

## V4.3.0 implementation candidate

V4.3.0 records SHA-256 and byte length in newly trained model metadata and verifies those values against exact deserialized bytes before inference, readiness, and evaluation. The Python package and Compose image candidate use `0.4.3`. Docker runtime and remote Actions remain unverified; no V4.3.0 tag or GitHub Release exists. See [the V4.3.0 validation record](releases/v4.3.0.md). All datasets and evaluation metrics remain synthetic.

## Historical tag inventory audit (2026-10-21)

The public GitHub Releases API returned zero releases; a Git tag is therefore not evidence of a published GitHub Release. Remote branch `main` matched local `main` at `c6f04704228571e1fb785bf95b36180964938262` before V4.3 changes. The repository's explicit release history and package-version commits support repairing two missing annotated tags; both are pointed to their historical release commits, never to current `HEAD`.

| Version | Intended/corresponding commit | Local tag | Remote tag | GitHub Release | Evidence / action |
| --- | --- | --- | --- | --- | --- |
| V3.0.0 | `180471113ff48ac73c2adfe8f7533111264b23b3` | `v3.0.0` | `v3.0.0` | None observed | Existing tag preserved. |
| V3.1.0 | `504af22dfca574b3bee2f905dcbd202605087288` | `v3.1.0` | `v3.1.0` | None observed | Changelog `[0.3.1]` and this commit's package bump/documented release identify the commit; created and pushed the missing annotated tag here. |
| V3.2.0 | `1a6a522703518c9655645a9d487d866d60e79553` | `v3.2.0` | `v3.2.0` | None observed | Existing tag preserved. |
| V3.3.0 | `138fea6d7ea428c084e255aa676dbebe28dc3ba1` | `v3.3.0` | `v3.3.0` | None observed | Existing tag preserved. |
| V3.4.0 | `a2d9c019bf449eafc42d691db9403c23d7ca0865` | `v3.4.0` | `v3.4.0` | None observed | Existing tag preserved. |
| V3.5.0 | `0d337aaa7b5d47448f2bc9dc2e8c53bc6b60c830` | `v3.5.0` | `v3.5.0` | None observed | Existing tag preserved. |
| V3.6.0 | `42b5d50813ac114d88c95ed41872c5ecd7333689` | `v3.6.0` | `v3.6.0` | None observed | Release commit, changelog `[0.3.6]`, and package bump identify the commit; created and pushed the missing annotated tag here. |
| V3.7.0 | `b76797ca32ad3be2fb7814994f8ab78a6356b256` | `v3.7.0` | `v3.7.0` | None observed | Existing tag preserved. |
| V3.8.0 | `e935b339540c88910d827d8c16e818effe2ec276` | `v3.8.0` | `v3.8.0` | None observed | Existing tag preserved. |
| V3.9.0 | `16128ff72025ff62e00359a606a1c3b32081cabd` | `v3.9.0` | `v3.9.0` | None observed | Existing tag preserved. |
| V4.0.0 | `78e0a3c9c3043f7a8db2b3b1c9cd1bd8d67770da` (implementation commit only) | Missing | Missing | None observed | Unresolved: resulting changelog still says V4.0 preparation/unreleased and package/image remain `0.3.8`; no release tag created. |
| V4.1.0 | `edc10f03b898217d7c0108115d0d111e53962d86` | `v4.1.0` | `v4.1.0` | None observed | Existing tag preserved; runtime/CI verification limitations remain documented. |
| V4.2.0 | `c6f04704228571e1fb785bf95b36180964938262` (candidate) | Missing | Missing | None observed | Documented candidate with Docker and Actions gates pending; no tag created. |
| V4.3.0 | `53fb0749996938f692cbe0adbdbd0d09211cc0f8` (candidate on `main`) | Missing | Missing | None observed | Docker runtime and remote Actions gates are pending; no stable tag created. |
| V4.4.0 | `99a849796b01c67e4ccfa4f9403ad53745d154f7` (candidate) | Missing | Missing | None observed | Docker, security, and remote Actions gates are pending; no stable tag created. |
| V5.0.0 | `0107e883f6fff6b5681d92b413f950694c002c48` | Missing | Missing | None observed | Implementation candidate; local checks pass, but Docker runtime and Actions were not confirmed at that release audit. |
| V5.1.0 | `789478211d3b99295cf54030d7690187413b4644` | `v5.1.0` | `v5.1.0` | `39861122427` passed | Published after local tests, lint/format/security checks, and remote Docker startup, API smoke, and persistence checks passed; see the V5.1.0 validation record. |

The 2026-10-21 audit fetched `origin` before implementation: remote `main` was `e1af8606133dc7cbca80eba681b97ef90c16f53a`; local and remote agreed. Existing remote tags `v3.0.0`–`v3.9.0` and `v4.1.0` were preserved and their peeled commit targets reviewed. Historical gaps `v4.0.0`, `v4.2.0`, `v4.3.0`, and `v4.4.0` remain unresolved candidates, not missing releases to synthesize: prior records explicitly document unreleased status or incomplete gates. No existing or older tags were moved. The public GitHub Releases API listed no releases and the Actions API had no workflow runs at audit time.

The 2026-10-22 audit began from clean `main` at `0107e883f6fff6b5681d92b413f950694c002c48` with no `v5.1.0` tag. After fixes and successful verification, release commit `789478211d3b99295cf54030d7690187413b4644` was pushed, tagged, and published as a stable GitHub Release. The final Actions run `39861122427` passed both jobs for that commit.
