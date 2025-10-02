# Data Scientist Challenge – Entel

Este repositorio contiene la resolución de la prueba práctica

## 📂 Estructura
- `data/` → dataset entregado
- `scripts/` → scripts reproducibles (part1_fill_answers.py`, `part2_train_model_xgb.py`)
- `outputs/` → métricas, gráficos y predicciones generada
- `respuestas.txt` → respuestas a las preguntas 1 a 4 de la primera parte
- `README.md` → este archiov

## Flujo de trabajo
1. **Parte 1 (EDA):** limpieza, análisis exploratorio y respuesta a preguntas descriptivas.
2. **Parte 2 (Modelado):**
   - División temporal: **train = junio**, **validación = julio**, **test = agosto**.
   - Modelos entrenados: **Logistic Regression, Random Forest, XGBoost**.
   - Selección del mejor modelo por **PR AUC en validación**.

## Resultados Primera Parte
### Pregunta 4 b.
Consideraría eliminar aquellas columnas con muy baja variabilidad (constantes), variables altamente correlacionadas con otras que aporten la misma información y columnas redundantes. Esto permitiría reducir dimensionalidad, simplificar el análisis y evitar problemas de multicolinealidad en los modelos.

### Pregunta 5.
Sí, existen diferencias en la distribución. En general, los clientes que se dan de baja tienden a recargar menos veces, y su distribución está más concentrada en valores bajos. En cambio, los clientes que no se dan de baja presentan una mayor dispersión, incluyendo casos de muchas recargas, lo que podría implicar que la frecuencia de recargas está asociada a la probabilidad de permanecer en la compañía.

## Resultados Segunda Parte
### Validación (julio)
| Modelo              | ROC AUC | PR AUC | F1   | Precisión | Recall |
|---------------------|---------|--------|------|-----------|--------|
| Logistic Regression | 0.77    | 0.11   | 0.27 | 0.23      | 0.32   |
| Random Forest       | 0.75    | 0.13   | 0.30 | 0.24      | 0.39   |
| XGBoost             | 0.80    | 0.17   | 0.34 | 0.27      | 0.46   |

### Test (agosto)
| Modelo elegido | ROC AUC | PR AUC | F1   | Precisión | Recall |
|----------------|---------|--------|------|-----------|--------|
| XGBoost        | 0.81    | 0.24   | 0.38 | 0.29      | 0.57   |

El modelo final elegido fue **XGBoost**, con mejor PR AUC y mayor recall, detectando más de la mitad de los clientes que efectivamente se dieron de baja.

## a. Métricas utilizadas
Dado que la clase churn es minoritaria, métricas como accuracy o ROC AUC pueden resultar optimistas y no reflejar la calidad en la clase positiva. Por lo mismo, se priorizaron métricas sensibles a la clase minoritaria y a la toma de decisiones operativas.

- **ROC AUC**: discriminación global (ya se sabe que no puede ser la métrica primaria ya que puede ser engañoso en datos desbalanceados).
- **PR AUC (Average Precision)**: métrica clave, enfocada en la clase minoritaria.
- **F1 Score, Precisión, Recall**: ayudan a balancear y entender los trade-offs.

## b. Posibles mejoras
- **Ingeniería de features**: variaciones mensuales, ratios, flags de caídas bruscas.
- **Modelado**: tuning de hiperparámetros, probar LightGBM/CatBoost.
- **Umbral**: ajustar según criterios de negocio (ej. recall mínimo).
- **Desbalance**: técnicas de oversampling/undersampling, ajuste fino de `scale_pos_weight`.

## c. Por qué estos modelos
El objetivo es construir un sistema que, mirando los datos pasados de un cliente (cuánto recarga, cuánto gasta, qué tanto usa el servicio, etc.), pueda predecir la categoría en la que probablemente caerá en el futuro (problema de clasificación).

- **Regresión Logística:** Modelo lineal, rápido e interpretable. Se usó como baseline para establecer un punto de comparación inicial.
- **Random Forest:** Ensamble de múltiples árboles de decisión entrenados con bagging. Captura relaciones no lineales y es más robusto que la regresión logística, pero puede requerir muchos árboles para alcanzar su mejor desempeño.
- **XGBoost:** Algoritmo de boosting que entrena árboles de forma secuencial corrigiendo errores previos. Es uno de los modelos más utilizados en problemas de churn por su capacidad de manejar desbalance y explotar interacciones complejas entre variables.

**XGBoost** resultó ser el modelo más adecuado para este dataset:
- En validación, XGBoost fue el modelo con mayor PR AUC (0.17), mostrando mejor balance en la identificación de bajas.
- En test, XGBoost alcanzó un ROC AUC de 0.81, un PR AUC de 0.24 (el doble que la Regresión Logística en validación) y un recall del 57%, logrando identificar a más de la mitad de los clientes que efectivamente se dieron de baja.
- Se observa que en los gráficos valid_xgboost_pr.png y valid_xgboost_roc.png XGBoost domina sobre los otros modelos en las curvas de precisión-recall y ROC.
