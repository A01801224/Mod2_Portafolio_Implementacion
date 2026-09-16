"""
visualizacion.py
Graficas para el reporte de la parte 3.

Reutiliza el estilo de la parte 2 (seaborn/matplotlib) y agrega las
graficas propias de este analisis: comparacion train/validacion/prueba,
el diagnostico de bias-varianza antes/despues, y el resultado del
GridSearchCV.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

CARPETA_RESULTADOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resultados"
)

sns.set_theme(style="whitegrid")


def _guardar(nombre_archivo):
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre_archivo)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Grafica guardada: {ruta}")


def graficar_matriz_confusion(resultados, titulo, nombre_archivo):
    """Identica a la de la parte 2."""
    matriz = [
        [resultados["TN"], resultados["FP"]],
        [resultados["FN"], resultados["TP"]]
    ]
    plt.figure(figsize=(5, 4))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicho 0", "Predicho 1"],
                yticklabels=["Real 0", "Real 1"])
    plt.title(titulo)
    _guardar(nombre_archivo)


def graficar_diagnostico_antes_despues(diagnostico_antes, diagnostico_despues):
    """
    Barras de F1 en entrenamiento vs validacion, antes y despues de la
    regularizacion, para visualizar como se cierra (o no) la brecha.
    """
    datos_grafica = pd.DataFrame([
        {"Momento": "Antes", "Conjunto": "Entrenamiento", "F1": diagnostico_antes["f1_entrenamiento"]},
        {"Momento": "Antes", "Conjunto": "Validacion", "F1": diagnostico_antes["f1_validacion"]},
        {"Momento": "Despues", "Conjunto": "Entrenamiento", "F1": diagnostico_despues["f1_entrenamiento"]},
        {"Momento": "Despues", "Conjunto": "Validacion", "F1": diagnostico_despues["f1_validacion"]},
    ])

    plt.figure(figsize=(7, 5))
    sns.barplot(data=datos_grafica, x="Momento", y="F1", hue="Conjunto",
                palette=["steelblue", "indianred"])
    plt.title("F1 en entrenamiento vs validacion, antes y despues de regularizar")
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.xlabel("")
    _guardar("diagnostico_antes_despues.png")


def graficar_resultados_grid_search(cv_results, top_n=15):
    """
    Barras horizontales con las top_n combinaciones de hiperparametros
    segun su F1 de validacion, para mostrar que tan sensible es el
    desempeno a cada configuracion.
    """
    tabla = pd.DataFrame(cv_results)
    tabla = tabla.sort_values("mean_test_score", ascending=False).head(top_n)
    etiquetas = [
        f"n={p['n_estimators']}, prof={p['max_depth']}, hoja={p['min_samples_leaf']}, {p['class_weight']}"
        for p in tabla["params"]
    ]

    plt.figure(figsize=(9, 7))
    sns.barplot(x=tabla["mean_test_score"], y=etiquetas, hue=etiquetas,
                palette="viridis", legend=False)
    plt.title(f"Top {top_n} combinaciones de hiperparametros (GridSearchCV, F1 en validacion)")
    plt.xlabel("F1 en validacion")
    plt.ylabel("")
    _guardar("grid_search_resultados.png")


def graficar_comparacion_final(metricas_antes, metricas_despues):
    """
    Barras comparando el modelo base y el modelo regularizado en las
    cinco metricas, evaluados sobre el conjunto de PRUEBA (evaluacion final).
    """
    metricas_nombres = ["accuracy", "precision", "recall", "specificity", "f1"]

    etiquetas = metricas_nombres * 2
    valores = ([metricas_antes[m] for m in metricas_nombres] +
               [metricas_despues[m] for m in metricas_nombres])
    modelos = (["Antes de regularizar"] * len(metricas_nombres) +
              ["Despues de regularizar"] * len(metricas_nombres))

    plt.figure(figsize=(10, 5))
    sns.barplot(x=etiquetas, y=valores, hue=modelos,
               palette=["steelblue", "seagreen"])
    plt.title("Comparacion final sobre el conjunto de prueba")
    plt.xlabel("")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.legend(title="")
    _guardar("comparacion_final.png")