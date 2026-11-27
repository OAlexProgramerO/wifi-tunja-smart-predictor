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

El verificador solo lee archivos locales y usa la biblioteca estándar. Sin `--save-history`, no inicia FastAPI/Streamlit, no abre SQLite ni accede a la red. Termina con código `0` para `VALID`, `1` para `INVALID` y `2` para `UNSUPPORTED`; imprime el estado y una razón breve. Archivos/manifiestos malformados, filtros/cantidades distintas y cambios en el digest de registros son inválidos. Versiones desconocidas de manifiesto/exportación, formatos o canonicalización no son compatibles.

### Digest de registros canónico

Primero se analizan los registros JSON y CSV a los mismos objetos de campos permitidos y privados. El arreglo ordenado `records` se serializa como UTF-8 con claves de objetos ordenadas recursivamente, separadores compactos, Unicode sin escapar y rechazo de números no finitos. Equivale a `json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`. Se conserva el orden del arreglo. El manifiesto guarda el SHA-256 hexadecimal en minúsculas de esos bytes e identifica esta regla como `json-sort-keys-compact-utf8-v1`. El escape CSV, saltos de línea e indentación JSON no afectan el digest; cambiar un registro o su orden sí.

El manifiesto incluye `manifest_schema_version`, `export_format`, `canonicalization`, `generated_at_utc`, `record_count`, `applied_filters` y `records_sha256`. Su fecha no se incluye en el digest de registros. La verificación también compara formato, filtros, esquema y cantidad con la exportación.

## Integridad y límites de datos

Los digests exportados conservan los valores y semántica de verificación de V8.6; las exportaciones no regeneran, actualizan ni reparan eventos o digests guardados. `VERIFIED`, `INVALID` y `UNVERIFIABLE` continúan siendo estados distintos. El digest local permite comprobar consistencia, pero no autenticidad criptográfica ni almacenamiento invulnerable a cambios; un operador privilegiado podría reescribir tanto el evento como su digest.

El exportador usa una lista explícita de campos y no serializa informes completos ni diccionarios de eventos sin filtrar. No exporta mensajes de chat, cuerpos de solicitudes, credenciales, datos personales ni cabeceras sensibles. Los informes solo cubren evaluaciones de políticas ML sintéticas; no incluyen los datos descriptivos oficiales de conectividad MinTIC.

Un resultado offline `VALID` solo comprueba que los registros de la exportación coinciden con el manifiesto bajo la canonicalización documentada. No es una firma, prueba de origen ni garantía contra modificaciones: quien pueda reemplazar ambos archivos también puede cambiar el digest del manifiesto.

## Historial local de verificaciones (V8.9)

Guardar es opcional. Añada `--save-history RUTA.sqlite` a una verificación para persistir una fila local:

```powershell
python scripts/verify_audit_export.py policy-evaluation-audit.json policy-evaluation-audit.json.manifest.json --save-history verification-history.sqlite
python scripts/verify_audit_export.py history list --store verification-history.sqlite --result VALID --limit 25 --offset 0
python scripts/verify_audit_export.py history list --store verification-history.sqlite --from-utc 2026-11-01T00:00:00Z --to-utc 2026-11-30T23:59:59Z
python scripts/verify_audit_export.py history summary --store verification-history.sqlite --from-utc 2026-11-01T00:00:00Z --result VALID
python scripts/verify_audit_export.py history show ID_VERIFICACION --store verification-history.sqlite
```

`history summary` acepta `--result`, `--from-utc` y `--to-utc`; agrega todas las filas coincidentes sin paginación. Sus campos JSON deterministas son `history_schema_version`, `filters`, `total_records`, `result_counts`, `format_counts`, `missing_format_count` y `date_range_utc` (`null` si no hay coincidencias). Los conteos por resultado siempre incluyen los tres estados. Los formatos nulos/desconocidos se informan aparte. Este resumen usa solo metadatos: las cantidades no miden calidad del modelo, rendimiento WiFi ni confiabilidad del servicio. Resumen/listado/detalle abren SQLite en modo de solo lectura.

El JSON del historial tiene esquema fijo, claves ordenadas y orden estable por fecha de verificación descendente y luego ID ascendente; `limit`/`offset` permiten paginar de forma estable. Cada fila contiene solo ID y fecha de verificación, nombre base y formato de exportación, versión de esquema del manifiesto, cantidad de registros, digest y resultado. El contenido de los registros exportados nunca se copia a SQLite ni a los informes. El comando normal sigue sin guardar historial salvo que se proporcione `--save-history`. Los errores de consulta del historial devuelven código `3`. Para listar se requiere un almacén existente; un almacén SQLite ausente, dañado o incompatible produce un error claro y nunca se recrea, reinicia ni migra silenciosamente. El historial contiene metadatos locales y no es un sistema de eventos de auditoría adicional.

El historial corresponde solo a evidencias de evaluación de políticas ML sintéticas y no incluye registros descriptivos oficiales MinTIC.

## Salida CSV del resumen (V8.11)

`history summary` continúa usando JSON determinista en la salida estándar de forma predeterminada. Seleccione CSV explícitamente; opcionalmente escriba cualquiera de los formatos en un archivo UTF-8:

```powershell
python scripts/verify_audit_export.py history summary --store verification-history.sqlite
python scripts/verify_audit_export.py history summary --store verification-history.sqlite --format csv
python scripts/verify_audit_export.py history summary --store verification-history.sqlite --result INVALID --from-utc 2026-11-01T00:00:00Z --format csv --output resumen-verificacion.csv
```

CSV usa UTF-8 sin marca BOM, finales de línea LF, columnas en orden fijo y `\n` como terminador de registro. Incluye una fila `summary`, tres filas `result` en orden `VALID`, `INVALID`, `UNSUPPORTED` y luego una fila `format` por cada formato disponible en orden alfabético. Los valores opcionales ausentes son celdas vacías; los conteos numéricos cero se escriben como `0`.

| Columna | Significado |
| --- | --- |
| `row_type` | Tipo de fila: `summary`, `result` o `format` |
| `history_schema_version` | Esquema del historial existente, en la fila resumen |
| `filter_result`, `filter_from_utc`, `filter_to_utc` | Filtros aplicados al resumen |
| `total_records` | Cantidad de verificaciones coincidentes con los filtros |
| `result` | Estado de cada fila de resultados |
| `result_count` | Cantidad para ese estado |
| `export_format` | Formato disponible en cada fila de formato |
| `format_count` | Cantidad para ese formato |
| `missing_format_count` | Filas coincidentes sin metadatos de formato |
| `date_range_from_utc`, `date_range_to_utc` | Marcas temporales mínima y máxima; vacías si no hay coincidencias |

JSON y CSV usan el mismo servicio y filtros; sus totales y conteos por grupo coinciden. El CSV solo contiene metadatos agregados de verificación; no incluye nombres de archivo, evidencias ni campos sensibles. Los conteos describen resultados locales, no autenticidad, calidad del modelo, rendimiento WiFi ni confiabilidad operativa. Un `--format` no soportado se rechaza; los errores de archivo de salida devuelven código `3`.

## Comprobación de integridad del resumen (V8.12)

Use `history check` para comparar el resumen existente con un recálculo sobre los mismos metadatos coincidentes. Es de solo lectura, funciona offline y acepta los mismos filtros de resultado/fecha:

```powershell
python scripts/verify_audit_export.py history check --store verification-history.sqlite
python scripts/verify_audit_export.py history check --store verification-history.sqlite --result VALID --from-utc 2026-11-01T00:00:00Z --to-utc 2026-11-30T23:59:59Z
```

El comando siempre genera JSON determinista. `CONSISTENT` (código `0`) significa que el total, conteos de resultados/formatos, formatos faltantes, rango de fechas, esquema y filtros coinciden entre el servicio de resumen y el recálculo. `INCONSISTENT` (código `1`) incluye hallazgos como diferencias de conteos o límites de fecha. `INDETERMINATE` (código `2`) significa que el almacén no está disponible o que las filas coincidentes no contienen metadatos suficientes/compatibles. Filtros inválidos y errores de uso CLI devuelven error no cero; el historial no se repara ni modifica.

Ejemplo de informe `CONSISTENT` (aquí se abrevian otros campos):

```json
{
  "integrity_report_schema_version": "1.0",
  "status": "CONSISTENT",
  "filters": {"result": null, "from_utc": null, "to_utc": null},
  "records_checked": 2,
  "findings": [],
  "summary": {"total_records": 2, "result_counts": {"VALID": 2}},
  "recomputed_summary": {"total_records": 2, "result_counts": {"VALID": 2}}
}
```

El arreglo `findings` contiene códigos legibles por máquina, campos afectados y valores observados/recálculados cuando existen. La comprobación solo detecta desacuerdos entre los metadatos locales y los agregados derivados. No valida exportaciones de origen, digests, autenticidad, no repudio ni almacenamiento invulnerable; un operador privilegiado puede cambiar la base local. Estos conteos locales no son métricas ML sintéticas, estadísticas oficiales MinTIC, calidad del modelo, rendimiento WiFi ni confiabilidad operativa.
