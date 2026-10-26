# V6.1.0 — Optimización de asignaciones en escenarios

V6.1 amplía el banco de escenarios existente con optimización determinista de asignaciones enteras. Usa el servicio predictor y el almacén de experimentos actuales; no entrena ni modifica modelos, no agrega dependencias de solver ni presenta recomendaciones operativas.

## Entradas y objetivos

La API `POST /experiments/optimize` recibe fecha y hora, entre una y doce zonas sintéticas, capacidad base, asignación de referencia opcional, límites enteros mínimos y máximos por zona, presupuesto total entero, objetivo y máximo de pasos. El objetivo ponderado exige una prioridad positiva para cada zona. El objetivo sin ponderación rechaza pesos para evitar ignorar supuestos silenciosamente.

Para cada zona `i`, el servicio recibe demanda sintética estimada fija `d_i`, capacidad base `c_i` y asignación entera `x_i`. El déficit es `q_i = max(0, d_i - c_i - x_i)`. Los objetivos disponibles son:

- `MINIMIZE_TOTAL_DEFICIT`: minimizar `sum(q_i)`.
- `MINIMIZE_WEIGHTED_DEFICIT`: minimizar `sum(w_i * q_i)`, con cada peso `w_i` positivo y explícito.

La demanda es distinta de la capacidad simulada asignada y permanece fija durante el cálculo. No se optimizan costo, latencia, rendimiento, energía, cobertura ni calidad de red porque no existen modelos válidos para esas variables.

## Método y optimalidad

El servicio puro comienza en los mínimos por zona y asigna repetidamente una unidad entera a la zona con mayor reducción marginal del déficit, con `zone_id` ascendente como desempate determinista. Se detiene cuando ninguna unidad mejora el objetivo, se agota el presupuesto, se alcanzan los máximos o se llega al límite de pasos configurado.

Para los objetivos aditivos disponibles, las mejoras marginales no aumentan, los límites por zona son independientes y existe un presupuesto compartido. Por ello, el método de heap demuestra un óptimo global si termina sin alcanzar el límite de pasos. Si queda asignación beneficiosa al llegar al límite, el candidato es factible pero se marca `search_limit_reached` y no se afirma optimalidad. Si la suma de mínimos excede el presupuesto, el problema es inviable y no se devuelve candidato seleccionado. Los empates se resuelven por orden lexicográfico del identificador de zona.

Los contadores de candidatos representan el estado inicial factible más cada estado tras una asignación unitaria aceptada; no enumeran todos los vectores posibles. Para un problema inviable se cuenta un caso inviable y cero para la búsqueda greedy factible. Los contadores describen el recorrido acotado.

## Ejemplo sintético ilustrativo

Suponga dos zonas con demandas fijas estimadas de 8 y 5 conexiones, capacidades base de 3 y 4, mínimos cero y un presupuesto compartido de 3 unidades enteras. Los déficits iniciales son 5 y 1. Con el objetivo sin ponderación, cada unidad reduce el déficit como máximo en uno; el desempate determinista asigna las tres unidades a la primera zona si su ID aparece primero al ordenar. Los déficits resultantes son 2 y 1 (total 3), frente al déficit inicial total 6. Es un ejemplo aritmético ilustrativo; no es una estimación real de Tunja ni una recomendación de despliegue.

## Persistencia, informes y procedencia

Las ejecuciones se guardan como campos JSON aditivos en el registro SQLite existente `capacity_scenario`, con `scenario_kind=allocation_optimization`; no hay otra base de datos ni migración. Consulte con `GET /experiments` y `GET /experiments/{id}`. Exporte JSON o Markdown mediante `GET /experiments/{id}/report?format=json|markdown` (esquema de informe 1.3). El informe incluye objetivo, restricciones y presupuesto, referencia, plan seleccionado o inviabilidad, demanda predicha fija y déficit por zona, códigos de razón, límites de búsqueda, procedencia de modelos/datos y limitaciones.

El Banco de escenarios en Streamlit ofrece un flujo de optimización junto al comparador what-if existente. Incluye controles explícitos en inglés y español, persistencia, recuperación y descarga JSON/Markdown.

## Límites de interpretación

Los datos, ubicaciones, predicciones y asignaciones son sintéticos. Los resultados dependen de las salidas registradas del modelo y no validan demanda pública real de WiFi, servicio, necesidades de infraestructura ni pronósticos reales en Tunja. La optimalidad matemática solo corresponde al objetivo sintético y las restricciones indicadas. La procedencia incompleta de modelos o datos se muestra como no evaluable; no se completa por inferencia.
