# V6.3.0 — Evaluación temporal y diagnósticos de cambio sintético

`POST /evaluation/temporal` ejecuta una evaluación adicional y acotada sobre el dataset sintético del proyecto. Clona los pipelines guardados de clasificación y regresión, ajusta cada copia únicamente con marcas temporales anteriores al bloque de prueba y presenta métricas y comparaciones de distribución de variables seleccionadas. Esta evaluación es independiente de la partición cronológica de prueba existente y de los diagnósticos expansivos guardados. No cambia la selección de modelos ni sus artefactos.

## Ventanas y métricas

El servicio ordena de forma estable las marcas temporales únicas, reserva la primera mitad como historial inicial de entrenamiento y divide el periodo restante en 2–5 bloques de prueba sin solapamiento. El entrenamiento se expande solo con filas anteriores. La solicitud permite definir mínimos de filas de entrenamiento y prueba (por defecto 100 y 20); las ventanas demasiado pequeñas y las ventanas de entrenamiento con una sola clase se marcan como omitidas. Se requiere al menos una marca temporal utilizable por ventana. Las marcas inválidas o variables no permitidas producen un error de validación.

Clasificación incluye accuracy, precision/recall/F1 de la clase positiva, ROC-AUC cuando existen ambas clases y probabilidades, y soporte/conteos de confusión. Regresión incluye MAE, RMSE, R² cuando está definido, sesgo y cuantiles del error absoluto. Una prueba con una sola clase puede evaluarse, pero ROC-AUC no se define. Los valores indefinidos o no finitos se serializan como `null`.

Para variables numéricas, el estadístico es la diferencia absoluta de medias dividida por esa diferencia más la desviación estándar agrupada; queda entre 0 y 1. Para variables categóricas se usa la distancia de variación total entre proporciones de categorías, también entre 0 y 1. Ambos métodos informan proporciones de faltantes y tamaños de muestra; menos de dos valores observados en cualquiera de los grupos produce `insufficient_samples`. Son comparaciones descriptivas, no umbrales de alerta ni pruebas de causalidad.

## API

Solicitud de ejemplo:

```http
POST /evaluation/temporal
Content-Type: application/json
```

```json
{"window_count": 3, "min_train_rows": 100, "min_test_rows": 20, "features": ["hour", "zone_type", "temperature_c"]}
```

La respuesta JSON versionada incluye versiones del proyecto y preprocesamiento, parámetros, límites y tamaños cronológicos, métricas disponibles, diagnósticos de variables, SHA-256 del dataset y de los artefactos de modelos, fecha de generación y limitaciones. Use `POST /evaluation/temporal?report_format=markdown` para obtener un resumen Markdown. Los límites son 2–5 ventanas, hasta 50.000 filas mínimas de entrenamiento, 10.000 de prueba y 12 variables únicas de entrada. Datasets más grandes o ajustes de modelos pueden tardar; no se crea un trabajo en segundo plano ni un almacén persistente.

## Dashboard e interpretación

En **Model Performance**, elija 2–5 ventanas y hasta 12 variables y ejecute la evaluación. El panel muestra métricas por ventana, comparaciones de variables y descargas JSON y Markdown. Los resultados se recalculan con el dataset sintético actual y los artefactos cargados; el JSON incluye huellas para reproducibilidad. Cambios en datos o marcas temporales cambian la identidad del informe.

Todas las filas, objetivos, ubicaciones y mediciones del proyecto son sintéticos. Una diferencia de distribución medida no demuestra deterioro del modelo, causalidad, cambio en vivo, menor calidad del servicio ni validez de pronósticos de demanda pública real de WiFi en Tunja. Estos diagnósticos no validan infraestructura municipal ni monitoreo de producción.
