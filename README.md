# Mod2_Portafolio_Implementacion

Portafolio de Implementación — Módulo 2 (Machine Learning), Inteligencia artificial avanzada
para la ciencia de datos (Gpo 601), Tecnológico de Monterrey.

Tres entregas sobre el mismo problema: predecir si una sesión de un sitio de e-commerce
terminará en compra, usando el [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(Sakar & Kastro, 2018, UCI Machine Learning Repository, licencia CC BY 4.0).

## Contenido

| Carpeta | Entregable | Algoritmo | Reporte |
|---|---|---|---|
| [`parte1_sin_framework/`](./parte1_sin_framework) | Implementación de una técnica de ML sin el uso de un framework | Árbol de decisión (ID3 adaptado a variables continuas, programado desde cero) | [PDF](./parte1_sin_framework/Árbol_de_decisión_sin_uso_de_framewrok.pdf) |
| [`parte2_con_framework/`](./parte2_con_framework) | Uso de framework o biblioteca de ML para la implementación de una solución | Random Forest (`scikit-learn`) | [PDF](./parte2_con_framework/Random_Forest_Retro2_Mod2.pdf) |
| [`parte3_desempenio_modelo/`](./parte3_desempenio_modelo) | Análisis y Reporte sobre el desempeño del modelo | Diagnóstico de bias/varianza y regularización sobre el Random Forest de la parte 2 | [PDF](./parte3_desempenio_modelo/Diagnostico_Regularizacion_Parte3.pdf) |

Cada carpeta es autocontenida: tiene su propio `data/`, `src/` y `resultados/`, y corre de forma
independiente con `python src/main.py`.

## Resumen de cada parte

**Parte 1 — Árbol de decisión sin framework.** Se implementó desde cero (solo con pandas para
manejo de datos y seaborn/matplotlib para graficar) un árbol de decisión tipo ID3 adaptado a
variables continuas: entropía y Gini como criterios de impureza, cortes binarios por umbral,
discretización por percentiles, y cuatro criterios de paro. Validado contra el ejercicio de Play
Tennis visto en clase. Resultado: accuracy 0.9031, F1 0.6571 sobre el conjunto de prueba.

**Parte 2 — Random Forest con scikit-learn.** Mismo dataset y preprocesamiento, ahora usando
`RandomForestClassifier`. Se compararon tres configuraciones (baseline, underfitting a propósito,
y óptima según experimentos de `n_estimators` y `max_depth`) para justificar la elección final.
Resultado: accuracy 0.9019, F1 0.6344 sobre el conjunto de prueba.

**Parte 3 — Diagnóstico y regularización.** Sobre la implementación de la parte 2, se hizo una
separación de tres vías (60% train / 20% validación / 20% prueba) para diagnosticar formalmente
bias, varianza y nivel de ajuste comparando desempeño en entrenamiento vs. validación. El modelo
base resultó bias bajo / varianza alta / overfit. Se aplicó `GridSearchCV` (120 combinaciones,
con un `PredefinedSplit` a la medida de la separación de tres vías) para regularizar, encontrando
`max_depth=15`, `min_samples_leaf=5`, `class_weight='balanced'`. El modelo regularizado pasó a
bias bajo / varianza media / fit, y mejoró el F1 en prueba de 0.6177 a 0.6695, con el mayor
avance en recall (0.5393 → 0.8246).

## Autor

Emilio Páez De la Mora - A01801224