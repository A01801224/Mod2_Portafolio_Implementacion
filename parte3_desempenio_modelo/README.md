# parte3_desempenio_modelo — Diagnóstico de Bias, Varianza y Regularización

Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo
(Portafolio Análisis). Inteligencia artificial avanzada para la ciencia de datos (Gpo 601),
Tecnológico de Monterrey.

Análisis de desempeño sobre el **Random Forest** de [`parte2_con_framework`](../parte2_con_framework):
separación en tres vías (train / validation / test), diagnóstico formal de bias, varianza y
nivel de ajuste, y regularización mediante búsqueda de hiperparámetros para mejorar el modelo.

## Dataset

Mismo dataset y preprocesamiento que las partes 1 y 2: [Online Shoppers Purchasing Intention
Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(Sakar & Kastro, 2018, UCI Machine Learning Repository, licencia CC BY 4.0), con las mismas 12
variables. A diferencia de las partes anteriores (80/20), aquí la separación es de **tres vías**:

| Conjunto | Proporción | Registros | Uso |
|---|---|---|---|
| Entrenamiento | 60% | 7,398 | Ajustar los árboles del modelo |
| Validación | 20% | 2,466 | Diagnosticar bias/varianza y elegir hiperparámetros |
| Prueba | 20% | 2,466 | Evaluación final única, nunca antes usada para decidir nada |

## Estructura de la carpeta

```
parte3_desempenio_modelo/
├── data/
│   └── online_shoppers_intention.csv
├── src/
│   ├── datos.py          # carga, preparacion y separacion train/validation/test
│   ├── diagnostico.py    # logica de diagnostico de bias, varianza y nivel de ajuste
│   ├── modelo.py          # entrenamiento del Random Forest y GridSearchCV con PredefinedSplit
│   ├── metricas.py        # matriz de confusion y metricas (sklearn.metrics)
│   ├── visualizacion.py   # graficas del reporte
│   └── main.py             # corre todo el analisis de punta a punta
├── resultados/             # graficas generadas al correr main.py
└── Diagnostico_Regularizacion_Parte3.pdf   # reporte con diagnostico y resultados
```

## Cómo correrlo

```bash
pip install pandas scikit-learn matplotlib seaborn
python src/main.py
```

No requiere notebook ni IDE, corre directo con el intérprete. La búsqueda de hiperparámetros
(120 combinaciones) tarda entre 15 y 20 segundos.

## Qué hace `main.py`

1. Carga el dataset y lo separa en train / validation / test (60/20/20).
2. Entrena el **modelo base** (`n_estimators=100`, `max_depth=None`, igual que `modelo_1` de la
   parte 2) usando solo el conjunto de entrenamiento, y lo diagnostica comparando su F1 en
   entrenamiento contra validación.
3. Busca los mejores hiperparámetros con `GridSearchCV`, usando un `PredefinedSplit` (en vez de
   k-fold) para que la búsqueda respete exactamente la separación train/validation: 120
   combinaciones sobre `n_estimators`, `max_depth`, `min_samples_leaf` y `class_weight`.
4. Entrena el **modelo regularizado** con los mejores hiperparámetros (otra vez solo con el
   conjunto de entrenamiento, para que el diagnóstico antes/después sea comparable) y lo
   vuelve a diagnosticar.
5. Reentrena ambos modelos con train+validación combinados y los evalúa **una sola vez** sobre
   el conjunto de prueba, que hasta ese punto no se había tocado.
6. Genera las 5 gráficas del reporte.

## Metodología de diagnóstico

El diagnóstico compara F1 en entrenamiento contra F1 en validación:

- **Bias** (según F1 de entrenamiento): bajo si ≥ 0.75, alto si < 0.45, medio entre ambos.
- **Varianza** (según la brecha F1 train − F1 val): baja si ≤ 0.05, alta si > 0.15, media entre
  ambos.
- **Nivel de ajuste**: bias alto → underfit; bias bajo/medio con varianza alta → overfit; bias
  bajo/medio con varianza baja/media → fit.

## Resultados

| | Antes de regularizar | Después de regularizar |
|---|---|---|
| Hiperparámetros | `n_estimators=100, max_depth=None` | `n_estimators=100, max_depth=15, min_samples_leaf=5, class_weight='balanced'` |
| F1 entrenamiento | 0.9991 | 0.7623 |
| F1 validación | 0.6230 | 0.6708 |
| Brecha train-val | +0.3761 | +0.0915 |
| Diagnóstico | bias bajo, varianza alta, **overfit** | bias bajo, varianza media, **fit** |
| F1 en prueba (final) | 0.6177 | **0.6695** |
| Recall en prueba | 0.5393 | **0.8246** |

La regularización redujo la brecha train-validación en 76% y mejoró el F1 final en prueba. La
mejora más grande está en recall (identifica 315 de 382 compradores reales, contra 206 antes),
gracias sobre todo a `class_weight='balanced'`, a costa de una caída en precision y accuracy.

El análisis completo, con metodología, figuras y conclusión, está en
[`Diagnostico_Regularizacion_Parte3.pdf`](./Diagnostico_Regularizacion_Parte3.pdf).

## Autor

Emilio Páez De la Mora - A01801224