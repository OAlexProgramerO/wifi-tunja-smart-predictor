# Exportación de evidencias de auditoría

V8.7 agrega exportación de solo lectura para la auditoría local de evaluaciones de políticas de V8.6. Las exportaciones reutilizan los mismos filtros exactos y rangos de fechas UTC inclusivos del historial, e incluyen todas las filas coincidentes, no solo la página visible del dashboard.

## API

- `GET /ml-experiments/policy-audit/export.json`
- `GET /ml-experiments/policy-audit/export.csv`

Ambos endpoints aceptan filtros opcionales `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc` y `created_to_utc`. Los límites de fecha deben ser marcas ISO ordenadas y con zona UTC. Rangos inválidos devuelven HTTP 422. Los resultados se ordenan por fecha descendente e ID de auditoría ascendente si las fechas coinciden. El orden de registros JSON es estable; la fecha de generación cambia naturalmente entre solicitudes.

El documento JSON incluye `export_schema_version`, `generated_at_utc`, `applied_filters`, `record_count` y `records`. Cada registro contiene una lista permitida de identificadores de auditoría/ejecución/evento, referencias a experimento y política si están disponibles, fecha, origen, resultado, digest local, estado/motivos de verificación y campos seleccionados de procedencia sintética. Se excluyen propiedades desconocidas del evento. Si el JSON está ausente o malformado, se representan los metadatos disponibles de la ejecución relacionada con estado `UNVERIFIABLE`, sin copiar el contenido corrupto.

El CSV usa columnas fijas. Su primera fila de datos tiene `row_type=metadata` e incluye esquema, fecha de generación, JSON canónico de filtros y cantidad de registros. Las siguientes filas tienen `row_type=audit_record`; una exportación vacía conserva el encabezado y la fila de metadatos.

Los filtros de auditoría del dashboard controlan ambas descargas. “Preparar exportaciones completas filtradas” solicita JSON y CSV usando solo los filtros; omite deliberadamente el `limit` y `offset` de la página visible.

## Integridad y límites de datos

Los digests exportados conservan los valores y semántica de verificación de V8.6; las exportaciones no regeneran, actualizan ni reparan eventos o digests guardados. `VERIFIED`, `INVALID` y `UNVERIFIABLE` continúan siendo estados distintos. El digest local permite comprobar consistencia, pero no autenticidad criptográfica ni almacenamiento invulnerable a cambios; un operador privilegiado podría reescribir tanto el evento como su digest.

El exportador usa una lista explícita de campos y no serializa informes completos ni diccionarios de eventos sin filtrar. No exporta mensajes de chat, cuerpos de solicitudes, credenciales, datos personales ni cabeceras sensibles. Los informes solo cubren evaluaciones de políticas ML sintéticas; no incluyen los datos descriptivos oficiales de conectividad MinTIC.
