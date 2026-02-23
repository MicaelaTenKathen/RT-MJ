# Resumen Ejecutivo — Sistema de Recomendacion B2B Quilmes

---

## 1. EDA: Hipótesis, Variables Predictivas y Feature Engineering

### 1.1 Datos y Alcance

---------------------------------------------------------------------------
Dimension                   Valor
---------------------------------------------------------------------------
Periodo                     24 Mayo – 31 Agosto 2022 (99 dias) 
Transacciones               280,788 
Ordenes                     45,547 
Clientes                    4,535 
SKUs (productos)            530 
Pares cliente-SKU unicos    117,157 
Esparcidad de la matriz     95.1 % 
---------------------------------------------------------------------------

**Calidad de datos**: Sin nulos en transacciones. Nulos menores en atributos (segmentoUnico 1.7 %, canal 0.3 %). Duplicados despreciables (40 exactos, 20 lógicos - diferencia en la columna 'INVOICE_DATE'). El 99.3 % de transacciones tienen atributos asociados. Se eliminaron los duplicados exactos y se validaron columnas requeridas antes de modelar.

### 1.2 Hipótesis planteadas a partir del EDA

El EDA se diseño para responder una pregunta central: **"Que variables del comportamiento de compra predicen mejor lo que un cliente va a comprar el mes siguiente?"**. A partir de la exploración, se formularon cuatro hipótesis:

**H1 — Frecuencia y recencia predicen recompra.**
El 47.6 % de los pares cliente-SKU tienen 2+ órdenes (tasa de recompra). De estos, el 28.2 % alcanzan 3+ y el 12.2 % alcanzan 5+ órdenes. Esto indica que un cliente que compró un SKU m+ultiples veces tiene alta probabilidad de volver a comprarlo. Las variables `frequency` (órdenes por par cliente-SKU) y `recency_days` (días desde la última compra) son las señales más directas.

**H2 — La cantidad comprada agrega información incremental.**
Un cliente que compra 100 cajas de un SKU tiene una relación más fuerte con ese producto que uno que compra 1 caja, aún con la misma frecuencia. La variable `total_quantity` (ITEMS_PHYS_CASES acumuladas) captura esta dimensión. Se aplica `log1p()` para amortiguar outliers (max 4,274 cajas).

**H3 — El canal determina el mix de productos relevantes.**
Los 11 canales muestran patrones de compra diferenciados: Mayorista promedia 3.2 SKUs/orden vs Entretenimiento 7.6. Un Kiosco compra productos distintos a un Restaurante. Si el modelo no conoce al cliente (cold-start), la popularidad de su canal es mejor predictor que la popularidad global.

**H4 — K dinámico supera a K fijo.**
La diversidad de compra varia entre clientes (1–155 SKUs únicos, media 25.8). Recomendar siempre 5 productos subrecomienda para clientes diversificados y sobrerecomienda para concentrados. La variable `SkuDistintosPromediosXOrden` del archivo de atributos permite personalizar K por cliente.

### 1.3 Feature Engineering aplicado

Las hipotesis se tradujeron en features concretas para los modelos:

------------------------------------------------------------------------------------
Feature              Descripción 
------------------------------------------------------------------------------------
`frequency`         COUNT DISTINCT(ORDER_ID) por (cliente, SKU) 
`recency_days`      MAX(INVOICE_DATE) por (cliente, SKU) vs fecha maxima 
`recency_boost`     Si la ultima compra fue < 30 dias +0.3 al valor de interaccion 
`total_quantity`    SUM(ITEMS_PHYS_CASES) por (cliente, SKU) 
`interaction`       Combinacion de frequency + recency_boost 
------------------------------------------------------------------------------------

**Elección de features** La matriz de interacción (95.1 % esparcidad) es la señal principal. En lugar de usar valores binarios (compro/no compro), se enriqueció con frecuencia logarítmica + boost de recencia. Esto permite que NMF y EASE distingan entre "compro 1 vez hace 3 meses" y "compro 10 veces, la última hace 5 días", sin que un único cliente con 98 órdenes domine la factorización (gracias al log1p-log(1+x)).

### 1.4 Visualizaciones generadas

Las siguientes visualizaciones fundamentan las hipótesis y se encuentran en `eda_outputs/`:

--------------------------------------------------------------------------------------------------------------------------------------------------
Gráfico                         Qué muestra                                                              Hipótesis que fundamenta 
--------------------------------------------------------------------------------------------------------------------------------------------------
`product_popularity.png`        Curva de Pareto: 17.7 % de SKUs genera 80 % de ordenes                   H3 — necesidad de fallback diferenciado 
`repurchase_distribution.png`   Distribución de frecuencia de recompra                                   H1 — frecuencia como predictor
`temporal_patterns.png`         Tendencia mensual creciente + órdenes por día de semana                  Justificación del periodo temporal 
`segment_channel_heatmap.png`   Heatmap segmento x canal (concentración de órdenes)                      H3 — canales diferenciados 
`customer_diversity.png`        Distribución de SKUs únicos por cliente                                  H4 — variabilidad justifica K dinámico 
--------------------------------------------------------------------------------------------------------------------------------------------------

### 1.5 Hallazgos clave del EDA

1. **Alta concentración**: El 17.7 % del catálogo (94 SKUs) genera el 80 % de las órdenes. El top 3 (SKUs 7038, 19088, 7651) acumula más de 34,000 órdenes. Se necesita fallback de popularidad para la cola larga.
2. **Recompra fuerte**: 47.6 % de pares cliente-SKU con 2+ órdenes. Frecuencia y recencia son predictivas.
3. **Esparcidad 95.1 %**: Se necesitan modelos que manejen matrices dispersas (NMF, EASE) y estrategias de fallback.
4. **Diversidad variable**: Clientes compran entre 1 y 155 SKUs únicos (media 25.8). Justifica K dinámico.
5. **Canales diferenciados**: SKUs/orden varia de 3.2 (Mayorista) a 7.6 (Entretenimiento). El fallback debe ser por canal.
6. **Tendencia creciente**: Junio 13,791 órdenes → Julio 14,999 → Agosto 16,711. La ventana expandible es apropiada.

---

## 2. Data Wrangling, Preprocesing y Modelado

### 2.1 Preprocesamiento realizado

1. **Carga y limpieza** (`eda/io.py`): Se cargan los dos datasets (transacciones y atributos de clientes). Se parsea `INVOICE_DATE` a datetime. Se eliminan 40 duplicados exactos y se validan las 5 columnas requeridas (`ACCOUNT_ID`, `SKU_ID`, `ORDER_ID`, `INVOICE_DATE`, `ITEMS_PHYS_CASES`).

2. **Calidad de datos** (`eda/io.py`): Se identifican y reportan nulos, duplicados lógicos (mismo ACCOUNT+ORDER+SKU: 20 casos), cantidades inválidas (0 casos), y cobertura transacciones-atributos (99.3 %). Los 156 clientes sin atributos se mantienen para entrenamiento pero reciben fallback global en recomendación.

3. **Construcción de la matriz de interacciones** (`models.py`): Se agrupan las transacciones por (ACCOUNT_ID, SKU_ID) calculando frequency, last_purchase, y total_quantity. Se construye una matriz sparse CSR de dimensión 4,535 x 530 con valores `log1p(frequency) + recency_boost`.

4. **Seed fija** (`np.random.seed(42)`): Garantiza reproducibilidad en la factorización NMF.

### 2.2 Supuestos del modelado

--------------------------------------------------------------------------------------------------------------------------------------------------
Supuesto                      Justificación 
--------------------------------------------------------------------------------------------------------------------------------------------------
**Reorder pattern**:          Se permiten recomendaciones de productos ya comprados. El 47.6 % de recompra indica que re-sugerir productos conocidos es válido en B2B.
**Interacción implícita**:    No hay ratings explícitos, se infiere interés del comportamiento. El dataset no tiene ratings. 
**Estacionariedad local**:    Los patrones recientes son representativos del futuro cercano. Con solo 99 dias no se puede modelar estacionalidad anual. Se asume que el comportamiento de Jun-Jul predice Ago. 
**Independencia de ítems**:   NMF y EASE no modelan secuencias ni canastas. Se modela la afinidad cliente-ítem, no el orden de compra ni la co-ocurrencia en una misma orden. 
--------------------------------------------------------------------------------------------------------------------------------------------------

### 2.3 Elección de modelos y justificación

Se eligieron tres modelos para comparar:

**Baseline Freq-Recency** — Se incluye como referencia obligatoria. En dominios con alta recompra, un modelo que simplemente rankea por "qué compro más y más recientemente" suele ser fuerte. Si los modelos avanzados no lo superan, no se justifica la complejidad adicional. Fórmula: `score = frequency + 0.3 * recency_score + 0.3 * log1p(quantity)`.
    **Referencias científicas:**
    - Bult, J. R., & Wansbeek, T. (1995). Optimal selection for direct mail. Marketing Science, 14(4), 378-394.
    - Hughes, A. M. (2005). Strategic database marketing. McGraw-Hill Pub. Co..
    - Manning, C. D. (2008). Introduction to information retrieval. Syngress Publishing,
    - Koren, Y. (2008, August). Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGK

**NMF (Non-negative Matrix Factorization)** — Factoriza la matriz de interacciones en dos matrices no-negativas W (usuarios x componentes) y H (componentes x items). Captura patrones latentes: clientes similares compran productos similares. Se aplica blending (`reconstruccion + 0.5 * raw`) para no perder la señal directa de compra histórica, ya que la reconstrucción pura tiende a suavizar demasiado en matrices muy esparsas (95.1 %).
    **Referencias científicas:**
    - Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. nature, 401(6755), 788-791.
    - Luo, X., Zhou, M., Xia, Y., & Zhu, Q. (2014). An efficient non-negative matrix-factorization-based approach to collaborative filtering for recommender systems. IEEE Transactions on Industrial informatics, 10(2), 1273-1284.
    - Koren, Y. (2008, August). Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 426-434).
    - Cacheda, F., Carneiro, V., Fernández, D., & Formoso, V. (2011). Comparison of collaborative filtering algorithms: Limitations of current techniques and proposals for scalable, high-performance recommender systems. ACM Transactions on the Web (TWEB), 5(1), 1-33.

**EASE (Embarrassingly Shallow Autoencoders, Steck 2019)** — Modelo ítem-ítem de forma cerrada: `B = -(X'X + lambda*I)^{-1}` con diagonal forzada a cero. No requiere iteraciones ni hiperparametros de aprendizaje. Captura relaciones ítem-ítem directamente. Se eligió por su simplicidad teórica y resultados competitivos en la literatura.
    **Referencias científicas:**
    - Steck, H. (2019, May). Embarrassingly shallow autoencoders for sparse data. In The World Wide Web Conference (pp. 3251-3257).

**Fallback de 2 niveles** — Común a los tres modelos. Cuando un cliente no tiene historial (cold-start) o su lista es incompleta, se complementa con: (1) productos más populares de su canal, y si no alcanza, (2) productos más populares globalmente. Esto garantiza que todo cliente recibe K recomendaciones completas.

---

## 3. Evaluacion: Métricas, Periodos y Resultados

### 3.1 Métricas de comparación

--------------------------------------------------------------------------------------------------------------------------------------------------
Métrica             Qué mide y relevancia
--------------------------------------------------------------------------------------------------------------------------------------------------
**Recall@K**        Proporcion de productos comprados que fueron recomendados. **Métrica principal**. En reposición B2B, no recomendar un producto que el cliente necesita tiene costo directo (venta perdida). Maximizar recall minimiza esas pérdidas.
**Precision@K**     Proporción de recomendaciones que fueron efectivamente compradas. Complementa al recall. Alta precisión = menos "ruido" en la lista. Relevante para no saturar al vendedor con sugerencias irrelevantes.
**MAP@K**           Precisión promedio ponderada por posición. Penaliza recomendaciones correctas que aparecen en posiciones bajas. Importa porque el vendedor ve primero los ítems del top del ranking.
**Hit Rate**        Proporción de clientes con al menos 1 acierto. Métrica de cobertura. Un hit rate del 95 % significa que el 95 % de los clientes recibe al menos una recomendación útil.
--------------------------------------------------------------------------------------------------------------------------------------------------

**Justificación de métrica principal** En un contexto B2B de reposición, el costo de no recomendar un producto que el cliente iba a comprar (falso negativo) es mayor que el costo de recomendar uno que no compra (falso positivo). El vendedor puede ignorar una sugerencia irrelevante, pero no puede adivinar un producto que el sistema no le mostro.
    **Referencia científica**
    - Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004). Evaluating collaborative filtering recommender systems. ACM Transactions on Information Systems (TOIS), 22(1), 5-53.

### 3.2 Periodo de evaluación y justificación

**Periodo del dataset**: Mayo–Agosto 2022 (99 días disponibles en el dataset).

**Estrategia de evaluación**: Expanding window con dos splits:

Split 1:  Train [Mayo–Junio]  →  Test [Julio]   
Split 2:  Train [Mayo–Julio]  →  Test [Agosto] 

**Motivo de elección del expanding window y no cross-validation**
- Los datos son temporales: usar datos futuros para predecir el pasado sería data leakage.
- Expanding window simula el escenario real de producción: cada mes se entrena con todo el histórico disponible y se predice el mes siguiente.
- Usar dos splits permite verificar que el modelo es **estable**: si gana en Julio pero pierde en Agosto, no es confiable.

### 3.3 Ajuste de hiperparámetros

Se realizó grid search con split temporal separado: Train (Mayo–Junio) → Validación (Julio), K=5. Total: 23 configuraciones evaluadas.

--------------------------------------------------------------------------------------------------------
Modelo                      Recall@10       Precision@10        Hit Rate            Configs evaluadas 
--------------------------------------------------------------------------------------------------------
Baseline Freq-Recency       0.2733          **0.5381**          0.8996                      9 
**NMF (k=20, blend=0.50)**  **0.3953**      0.4532              **0.9327**                  9 
EASE (lambda=50)            0.3278          0.5226              0.4666                      5 
--------------------------------------------------------------------------------------------------------


**Mejores hiperparámetros encontrados**:
- Baseline: peso_recencia (recency_weights) = 0.3, peso_cantidad (quantity_weights) = 0.3
- NMF:      n_componentes (n_components) = 20, peso_blending (blend_weight) = 0.50
- EASE:     lambda_reg = 50

**Observaciones del tuning**:
- **NMF supera al Baseline en recall por +12.2 pp** (0.3953 vs 0.2733), demostrando que los factores latentes capturan patrones que la frecuencia y la recencia solas no detectan.
- Baseline obtiene la mejor precision (0.5381), lo que indica que cuando recomienda, acierta más — pero recomienda menos productos correctos en total (menor recall).
- EASE queda en segundo lugar en recall (0.3679) y muestra poca sensibilidad al lambda_reg (rango 50–300 varia recall solo en 0.002).
- Los tres modelos logran hit rates >89 %, indicando que casi siempre aciertan al menos un producto.

### 3.4 Evaluación final

Los tres modelos se evaluaron en ambos splits con **K dinámico** por cliente. Resultados:

------------------------------------------------------------------------------------------------------------------------------
Modelo      Recall Jul  Recall Ago  **Recall Prom**     Prec Jul    Prec Ago    MAP Jul     MAP Ago     Hit Jul     Hit Ago 
------------------------------------------------------------------------------------------------------------------------------
Baseline    0.3498      0.3698      0.3598              0.4925      0.5594      0.4478      0.5238      0.9213      0.9478 
**NMF**     **0.3568**  **0.3786**  **0.3677**          0.5115      0.5771      0.4742      0.5458      0.9142      0.9462 
EASE        0.3149      0.3278      0.3213              0.4686      0.5226      0.4124      0.4666      0.8882      0.9221 
------------------------------------------------------------------------------------------------------------------------------

**Ganador: NMF (k=20, blend=0.5) con recall promedio = 0.3677**

NMF gana en ambos splits (Julio y Agosto), confirmando su estabilidad. También lidera en precisión (0.5115 / 0.5771) y MAP (0.4742 / 0.5458), lo que indica que no solo captura más productos correctos, sino que los ubica mejor en el ranking.

Todos los modelos mejoran de Julio a Agosto en todas las métricas, lo cual es esperable: el Split 2 tiene más datos de entrenamiento (3 meses vs 2).

### 3.5 Validación de H4: K Fijo vs K Dinámico

Se comparó K fijo = 5 (derivado del EDA: media 3.0 SKUs/orden x 1.5) vs K dinámico personalizado, usando NMF sobre el split de Agosto:

--------------------------------------------------------------------------------
Métrica         K Fijo = 5      K Dinamico (3-12)       Delta 
--------------------------------------------------------------------------------
**Recall**      0.2927          0.3786                  **+0.0859 (+29.3 %)** 
Precision       0.6237          0.5771                  -0.0466 (-7.5 %) 
Hit Rate        0.9357          0.9462                  +0.0105 (+1.1 %) 
--------------------------------------------------------------------------------

**H4 validada**: El K dinámico mejoró el recall en **29.3 %** respecto al K fijo. La caída de precisión (-7.5 %) es esperable y aceptable: al recomendar más items (K promedio = 8 vs 5), la proporción de aciertos baja, pero se capturan muchos más productos que el cliente efectivamente compró. El hit rate tambien mejora ligeramente (+1.1 %).

---

## 4. Output de Producción

### 4.1 Output del sistema

`3-final_evaluation.py` produce el archivo `model_outputs/recommendations_production.csv` con recomendaciones listas para consumo del modelo ganador (NMF).

### 4.2 Esquema del CSV

------------------------------------------------------------------------------------------------
Columna             Tipo        Descripción
------------------------------------------------------------------------------------------------
`scoring_date`      date        Día siguiente al último dato de entrenamiento (2022-09-01) 
`ACCOUNT_ID`        int         Identificador único del cliente 
`SKU_ID`            int         Producto recomendado 
`rank`              int         Posición en el ranking (1 = más relevante) 
`score`             float       Score del modelo NMF (mayor = más confianza) 
`k_recommended`     int         Cantidad de recomendaciones generadas para este cliente 
`model`             string      Modelo usado: `nmf_(k=20,_blend=0.5)` 
`generation_ts`     datetime    Timestamp UTC de generación 
------------------------------------------------------------------------------------------------

### 4.3 Número de recomendaciones por cliente

Cada cliente recibe entre **3 y 12 recomendaciones**. La cantidad exacta (K) se define individualmente.

**Criterio para definir K**: Se usa el atributo `SkuDistintosPromediosXOrden` (SKUs distintos promedio por orden de ese cliente), multiplicado por 1.5 y acotado al rango [3, 12]:

```
K = clip(round(SkuDistintosPromediosXOrden * 1.5), min=3, max=12)
```

**Motivo del criterio**

- **El factor 1.5** busca ligeramente recomendar más de lo que el cliente compra habitualmente por orden, dejando margen para descubrimiento sin saturar la lista.
- **Mínimo 3**: Garantiza que incluso clientes de baja diversidad reciban un mínimo útil de sugerencias. Con menos de 3 el vendedor no tiene opciones.
- **Máximo 12**: Evita listas excesivamente largas.
- **Basado en el comportamiento real del cliente**: No es un número arbitrario global.

### 4.4 Estadísticas del output generado

---------------------------------------------------------------------------------
Dimension                               Valor 
---------------------------------------------------------------------------------
Total de filas (recomendaciones)        35,116 
Clientes cubiertos                      4,400 (100 % del archivo de atributos) 
K minimo usado                          3 
K maximo usado                          12 
K promedio                              8.0 
K mediana                               8 
Scoring date                            2022-09-01 
Modelo                                  NMF (k=20, blend=0.5) 
----------------------------------------------------------------------------------

**Distribución de K en el output**:

-------------------
K       Clientes 
-------------------
 3        287 
 4        391 
 5        415 
 6        557 
 7        395 
 8        439 
 9        380 
10        320 
11        215 
12      1,001 
-------------------

El K=12 es el más frecuente (1,001 clientes), correspondiente a clientes de alta diversidad (SkuDistintosPromediosXOrden >= 8). La distribución confirma que el K dinámico se adapta al perfil de cada cliente.

### 4.5 Cobertura y cold-start

- Se generan recomendaciones para los **4,400 clientes** del archivo de atributos, no solo los que tienen historial de compra.
- Clientes sin historial (cold-start) reciben recomendaciones via fallback: primero productos populares de su canal, luego populares globales.
- El CSV se ordena por `(ACCOUNT_ID, rank)` para facilitar consulta y carga en sistemas downstream.

---

## 5. Arquitectura del Sistema

```
eda/
  io.py              Carga y limpieza de datos
  quality.py         Análisis estadísticos
  plots.py           Generación de gráficos
  report.py          Reporte EDA en TXT

models.py            Modelos (Baseline, NMF, EASE) + utils (matriz, fallback, K dinámico)

tuning/
  eval.py            Split de validación para tuning
  baseline.py        Grid search Baseline
  nmf.py             Grid search NMF
  ease.py            Grid search EASE
  report.py          Reportes de tuning

1-eda-analysis.py            Ejecuta el EDA completo
2-hyperparameter-tuning.py   Ejecuta el ajuste de hiperparámetros
3-final_evaluation.py        Evaluación final + output de producción

test.py                      Splits temporales, métricas, comparación rápida (Agosto)
output_stats.py              Genera estadísticas del CSV output
```

Los scripts se ejecutan en orden numérico: primero el EDA (1), luego el tuning (2), finalmente la evaluación y generación del output (3).

---

## 6. Limitaciones

1. **Periodo corto** (99 días): No captura estacionalidad anual. Con 12+ meses se podrían detectar ciclos de compra estacionales y mejorar las predicciones. Así también, con un mayor número de datos, se podrían aplicar modelos más complejos.
2. **Cold-start limitado**: Clientes sin historial reciben solo fallback de popularidad. Se podría explorar content-based filtering usando los atributos del cliente (BussinessSegment, nse, concentración) para recomendar lo que compran clientes similares.
3. **Escalabilidad de EASE**: Para catálogos grandes (>10K SKUs) habría que mejorar el modelo.
