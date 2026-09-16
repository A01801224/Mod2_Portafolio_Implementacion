# Retro1_Mod2

Momento de Retroalimentación: Módulo 2 — Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

**Matrícula:** A01801224
**Materia:** Módulo 2 – Aprendizaje Máquina (Prof. Uresti)

## Descripción

Implementación manual de un **árbol de decisión** (ID3 con entropía y ganancia de información) para clasificación binaria.
El cálculo de impureza, la búsqueda del mejor corte, la construcción
recursiva del árbol y la predicción están programados desde cero.

Se implementaron los dos criterios de impureza (entropía y Gini) para
poder compararlos, y el reporte incluye dos experimentos: el efecto de
la profundidad máxima sobre el sobreajuste, y entropía contra Gini.

No se usa ninguna biblioteca de aprendizaje máquina. Pandas se usa únicamente
para leer y manipular la tabla de datos, y seaborn/matplotlib únicamente para
generar las gráficas del reporte.

## Dataset

**Online Shoppers Purchasing Intention Dataset** (UCI Machine Learning Repository).

12,330 sesiones de un sitio de comercio electrónico, cada una perteneciente a un
usuario distinto a lo largo de un periodo de un año. El objetivo es predecir si
la sesión terminó en compra (columna `Revenue`).

- Variables usadas: las 10 numéricas del dataset, más `Weekend` y `VisitorType`
  codificadas manualmente como 0/1.
- Variables descartadas: `Month`, `OperatingSystems`, `Browser`, `Region` y
  `TrafficType`, por ser códigos categóricos sin orden real. Un corte del tipo
  "Browser < 7" no tendría significado.
- Entrenamiento: 80% de las sesiones. Prueba: 20%.
- La separación usa semilla fija (42) para que los resultados sean reproducibles.
- El dataset no tiene valores faltantes.

Las clases están desbalanceadas: la mayoría de las sesiones no terminan en compra.
Por eso el reporte no se limita a accuracy e incluye recall y F1.

**Cita:** Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention
Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
Licencia Creative Commons Attribution 4.0 International (CC BY 4.0).

## Estructura

```
.
├── src/
│   ├── datos.py                  # carga, codificación y separación train/test
│   ├── metricas.py               # matriz de confusión y métricas, calculadas a mano
│   ├── visualizacion.py          # gráficas del reporte
│   ├── arbol.py                  # árbol de decisión implementado desde cero
|   └── validacion_playtennis.py  # reproduce el ejercicio de clase
│   └── main.py                   # experimentos, modelo final y predicciones
├── data/
│   └── online_shoppers_intention.csv
├── resultados/            # gráficas generadas y reporte.pdf
├── requirements.txt
└── README.md
```

## Cómo ejecutar

Requiere Python 3.9 o superior.

```bash
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activarlo
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac o Linux

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Correr el programa
python src/main.py
```

Cada módulo también se puede ejecutar por separado para probarlo:

```bash
python src/datos.py                    # resumen del dataset
python src/arbol.py                    # pruebas de las formulas del arbol
python src/validacion_playtennis.py    # reproduce el ejercicio de clase
python src/metricas.py                 # verificación de las fórmulas
python src/visualizacion.py            # genera las gráficas exploratorias
```

## Qué se implementó a mano

- Cálculo de impureza: entropía y Gini
- Ganancia de información
- Discretización de variables continuas en cortes candidatos por percentiles
- Búsqueda del mejor corte entre todas las variables
- Construcción recursiva del árbol y cuatro criterios de paro (nodo puro,
  mínimo de muestras, profundidad máxima, sin ganancia posible)
- Predicción recorriendo el árbol
- Codificación de las variables binarias y separación entrenamiento/prueba
- Matriz de confusión, accuracy, precision, recall, specificity y F1

Las fórmulas están validadas contra el ejercicio de Play Tennis visto en
clase: la implementación reproduce exactamente Entropy(S) = 0.9403 y
Gain(S, Outlook) = 0.2467.

## Resultados

Ver `resultados/reporte.pdf`.
