# Retro2_Mod2 — Random Forest

Módulo 2: Uso de framework o biblioteca de aprendizaje máquina para la implementación
de una solución. Inteligencia artificial avanzada para la ciencia de datos (Gpo 601),
Tecnológico de Monterrey.

Implementación de un **Random Forest** con `scikit-learn` para predecir si una sesión
de un sitio de e-commerce terminará en compra. Es la continuación con framework del
árbol de decisión programado a mano en [Retro1_Mod2](https://github.com/A01801224/Retro1_Mod2).

## Dataset

[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(Sakar & Kastro, 2018, UCI Machine Learning Repository, licencia CC BY 4.0).
12,330 sesiones de un sitio de e-commerce. La clase a predecir es `Revenue`
(si la sesión terminó en compra), fuertemente desbalanceada: 84.5% no compra
vs 15.5% sí compra.

Se usan las mismas 12 variables que en la parte 1 (10 numéricas + Weekend y
VisitorType codificadas a mano). El resto de las columnas se descartan por ser
categóricas sin orden real. El preprocesamiento es idéntico entre ambos repos
para que los resultados sean comparables.

## Estructura del repositorio

```
Retro2_Mod2/
├── data/
│   └── online_shoppers_intention.csv
├── src/
│   ├── datos.py          # carga, preparacion y separacion del dataset
│   ├── modelo.py          # configuracion y entrenamiento del Random Forest
│   ├── metricas.py        # matriz de confusion y metricas (sklearn.metrics)
│   ├── visualizacion.py   # graficas del reporte (seaborn/matplotlib/plot_tree)
│   └── main.py             # corre todo el pipeline de punta a punta
├── resultados/             # graficas generadas al correr main.py
└── Random_Forest_Retro2_Mod2.pdf   # reporte con resultados y analisis
```

## Cómo correrlo

```bash
pip install pandas scikit-learn matplotlib seaborn
python src/main.py
```

No requiere notebook ni IDE, corre directo con el intérprete. Al terminar,
las gráficas quedan guardadas en `resultados/`.

## Qué hace `main.py`

1. Carga y resume el dataset (distribución de clases, piso de referencia).
2. Entrena `modelo_1`: baseline con los valores por defecto de scikit-learn
   (`n_estimators=100`, `max_depth=None`).
3. Experimento 1: efecto de `n_estimators` (5 a 300 árboles) sobre F1 en
   entrenamiento y prueba.
4. Experimento 2: efecto de `max_depth` (2 a sin límite) sobre F1 en
   entrenamiento y prueba.
5. Entrena `modelo_2` (underfitting a propósito: `n_estimators=5`,
   `max_depth=2`) y `modelo_3` (configuración óptima según los experimentos:
   `n_estimators=100`, `max_depth=10`).
6. Compara los tres modelos en las cinco métricas.
7. Imprime predicciones de ejemplo en consola usando `modelo_3`.
8. Genera las 9 gráficas del reporte.

## Resultados (conjunto de prueba, 2,466 registros)

| Modelo | n_estimators | max_depth | Accuracy | F1 |
|---|---|---|---|---|
| modelo_1 (baseline) | 100 | None | 0.8966 | 0.6177 |
| modelo_2 (underfitting) | 5 | 2 | 0.8609 | 0.2222 |
| **modelo_3 (óptimo)** | 100 | 10 | **0.9019** | **0.6344** |

`modelo_3` es el modelo final: mejora al baseline en las cinco métricas al
limitar la profundidad de cada árbol, lo que reduce el sobreajuste sin
necesitar más árboles. `PageValues` es, por mucho, la variable más importante
(39% de `feature_importances_`), consistente con el árbol de la parte 1.

El análisis completo, con matriz de confusión, gráficas de los experimentos
y conclusión, está en
[`Random_Forest_Retro2_Mod2.pdf`](./Random_Forest_Retro2_Mod2.pdf).

## Autor

Emilio Páez De la Mora - A01801224