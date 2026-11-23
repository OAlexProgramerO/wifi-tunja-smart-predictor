# Exportación de evidencias de auditoría

V8.7 agrega exportación de solo lectura para la auditoría local de evaluaciones de políticas de V8.6. Las exportaciones reutilizan los mismos filtros exactos y rangos de fechas UTC inclusivos del historial, e incluyen todas las filas coincidentes, no solo la página visible del dashboard.

## API

- `GET /ml-experiments/policy-audit/export.json`
- `GET /ml-experiments/policy-audit/export.csv`
- `GET /ml-experiments/policy-audit/export.manifest.json?export_format=json|csv`

Ambos endpoints aceptan filtros opcionales `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc` y `created_to_utc`. Los límites de fecha deben ser marcas ISO ordenadas y con zona UTC. Rangos inválidos devuelven HTTP 422. Los resultados se ordenan por fecha descendente e ID de auditoría ascendente si las fechas coinciden. El orden de registros JSON es estable; la fecha de generación cambia naturalmente entre solicitudes.

El documento JSON incluye `export_schema_version`, `generated_at_utc`, `applied_filters`, `record_count` y `records`. Cada registro contiene una lista permitida de identificadores de auditoría/ejecución/evento, referencias a experimento y política si están disponibles, fecha, origen, resultado, digest local, estado/motivos de verificación y campos seleccionados de procedencia sintética. Se excluyen propiedades desconocidas del evento. Si el JSON está ausente o malformado, se representan los metadatos disponibles de la ejecución relacionada con estado `UNVERIFIABLE`, sin copiar el contenido corrupto.

El CSV usa columnas fijas. Su primera fila de datos tiene `row_type=metadata` e incluye esquema, fecha de generación, JSON canónico de filtros y cantidad de registros. Las siguientes filas tienen `row_type=audit_record`; una exportación vacía conserva el encabezado y la fila de metadatos.

Los filtros de auditoría del dashboard controlan ambas descargas. “Preparar exportaciones completas filtradas” solicita JSON y CSV usando solo los filtros; omite deliberadamente el `limit` y `offset` de la página visible.

## Manifiesto offline y verificación CLI (V8.8)

El dashboard genera un manifiesto separado para cada respuesta JSON/CSV descargada y ofrece los cuatro archivos: exportación JSON, manifiesto JSON, exportación CSV y manifiesto CSV. El endpoint API de manifiesto usa los mismos filtros y permite `export_format=json` o `export_format=csv`. Si solicita por separado una exportación API y su manifiesto, use los mismos filtros; si los registros coincidentes cambian entre solicitudes, la verificación detectará la diferencia. Los manifiestos del dashboard se calculan directamente desde los bytes descargados.

Ejemplos:

```powershell
python scripts/verify_audit_export.py policy-evaluation-audit.json policy-evaluation-audit.json.manifest.json
python scripts/verify_audit_export.py policy-evaluation-audit.csv policy-evaluation-audit.csv.manifest.json
```

El verificador solo lee archivos locales y usa la biblioteca estándar. No inicia FastAPI/Streamlit, no abre SQLite ni accede a la red. Termina con código `0` para `VALID`, `1` para `INVALID` y `2` para `UNSUPPORTED`; imprime el estado y una razón breve. Archivos/manifiestos malformados, filtros/cantidades distintas y cambios en el digest de registros son inválidos. Versiones desconocidas de manifiesto/exportación, formatos o canonicalización no son compatibles.

### Digest de registros canónico

Primero se analizan los registros JSON y CSV a los mismos objetos de campos permitidos y privados. El arreglo ordenado `records` se serializa como UTF-8 con claves de objetos ordenadas recursivamente, separadores compactos, Unicode sin escapar y rechazo de números no finitos. Equivale a `json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`. Se conserva el orden del arreglo. El manifiesto guarda el SHA-256 hexadecimal en minúsculas de esos bytes e identifica esta regla como `json-sort-keys-compact-utf8-v1`. El escape CSV, saltos de línea e indentación JSON no afectan el digest; cambiar un registro o su orden sí.

El manifiesto incluye `manifest_schema_version`, `export_format`, `canonicalization`, `generated_at_utc`, `record_count`, `applied_filters` y `records_sha256`. Su fecha no se incluye en el digest de registros. La verificación también compara formato, filtros, esquema y cantidad con la exportación.

## Integridad y límites de datos

Los digests exportados conservan los valores y semántica de verificación de V8.6; las exportaciones no regeneran, actualizan ni reparan eventos o digests guardados. `VERIFIED`, `INVALID` y `UNVERIFIABLE` continúan siendo estados distintos. El digest local permite comprobar consistencia, pero no autenticidad criptográfica ni almacenamiento invulnerable a cambios; un operador privilegiado podría reescribir tanto el evento como su digest.

El exportador usa una lista explícita de campos y no serializa informes completos ni diccionarios de eventos sin filtrar. No exporta mensajes de chat, cuerpos de solicitudes, credenciales, datos personales ni cabeceras sensibles. Los informes solo cubren evaluaciones de políticas ML sintéticas; no incluyen los datos descriptivos oficiales de conectividad MinTIC.

Un resultado offline `VALID` solo comprueba que los registros de la exportación coinciden con el manifiesto bajo la canonicalización documentada. No es una firma, prueba de origen ni garantía contra modificaciones: quien pueda reemplazar ambos archivos también puede cambiar el digest del manifiesto.
