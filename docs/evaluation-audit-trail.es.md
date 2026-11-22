# Trazabilidad de auditoría e integridad de evaluaciones

V8.6 guarda un evento de auditoría local en la misma transacción SQLite que cada nueva ejecución de evaluación de política. El evento relaciona el ID de auditoría con la ejecución y registra experimento, ID/versión exactos de política, fecha UTC, origen de ejecución de la aplicación (`api`), resultado, versión del proyecto, tarea, huella del dataset sintético y partición/ventana de evaluación. No copia mensajes de chat, cuerpos de solicitud, credenciales, datos personales ni cabeceras sensibles. Solo describe evaluaciones ML sintéticas y permanece separado de los registros oficiales MinTIC.

## Codificación canónica y digest

El JSON del evento se serializa como UTF-8 sin marca BOM, con claves de objetos ordenadas recursivamente, sin espacios insignificantes, caracteres Unicode sin escapar y números no finitos rechazados. En Python equivale a:

```python
json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

El digest es SHA-256 hexadecimal en minúsculas de esos bytes UTF-8. El digest cubre el objeto del evento, no la fila de base de datos ni el informe de ejecución enlazado. La verificación también compara columnas indexadas y referencias del evento con la ejecución y el informe guardados.

## API de solo lectura

- `GET /ml-experiments/policy-runs/{run_id}/audit/verify` verifica el evento de auditoría de una ejecución. Si falta, devuelve `UNVERIFIABLE`; no reconstruye eventos históricos.
- `GET /ml-experiments/policy-audit/{audit_id}/verify` verifica por ID de auditoría y comprueba la ejecución relacionada.
- `GET /ml-experiments/policy-audit` devuelve eventos con `limit` (1–100), `offset` (hasta 100000) y filtros opcionales `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc` y `created_to_utc`. Los límites de fecha son inclusivos y usan marcas ISO con zona UTC. Cada elemento incluye el resultado de verificación.

Estados de verificación:

- `VERIFIED`: esquema admitido, digest local coincidente, campos indexados consistentes y referencias de ejecución/informe coherentes.
- `INVALID`: se detectó una diferencia de digest o un registro relacionado modificado, ausente o incoherente.
- `UNVERIFIABLE`: el evento falta, no se puede leer o usa un esquema no compatible, por lo que no es posible verificarlo de forma fiable.

El dashboard presenta estados, detalles, filtros y verificación por ID de ejecución. Las ejecuciones V8.5 son anteriores a estos eventos y por eso aparecen como `UNVERIFIABLE`; no se cambian sus resultados ni versiones de política.

## Seguridad y límites

Esto es una comprobación local de consistencia, no una firma digital ni una prueba de origen. No vuelve SQLite invulnerable a modificaciones. Un operador con privilegios sobre la base puede reescribir tanto el evento como su digest o reemplazar toda la base. No se usa servicio de auditoría externo, llave ni dependencia paga. `VERIFIED` solo significa que el digest local actual y los datos enlazados son consistentes con la codificación documentada.

V8.7 agrega descargas JSON/CSV filtradas de todo el historial de auditoría coincidente. Consulte la [guía de exportación de evidencias](evaluation-audit-export.es.md) para conocer campos, metadatos y orden.
