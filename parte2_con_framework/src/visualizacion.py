"""
visualizacion.py
Graficas para el reporte, hechas con seaborn, matplotlib y sklearn.tree.

Estas bibliotecas solo se usan para visualizar, no para modelar
(la unica excepcion es plot_tree, que es parte de sklearn pero solo
dibuja un arbol ya entrenado, no participa en el entrenamiento).
Todas las graficas se guardan como png en la carpeta resultados/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree

from datos import VARIABLES_NUMERICAS, COLUMNA_CLASE, VARIABLES

# Carpeta donde se guardan las imagenes, relativa a este archivo
CARPETA_RESULTADOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resultados"
)

sns.set_theme(style="whitegrid")


def _guardar(nombre_archivo):
    """Guarda la figura actual en resultados/ y limpia la figura."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    ruta = os.path.join(CARPETA_RESULTADOS, nombre_archivo)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"Grafica guardada: {ruta}")


def graficar_distribucion_clases(datos):
    """Barras con cuantas sesiones terminaron en compra. Muestra el desbalance.
    Identica a la parte 1: el dataset no cambio, solo el modelo."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=datos, x=COLUMNA_CLASE, hue=COLUMNA_CLASE,
                  palette=["steelblue", "seagreen"], legend=False)
    plt.title("Distribucion de clases")
    plt.xlabel("Compra (0 = No, 1 = Si)")
    plt.ylabel("Cantidad de sesiones")
    _guardar("distribucion_clases.png")


def graficar_histogramas(datos):
    """Un histograma por variable numerica, separado por clase.
    Identica a la parte 1."""
    fig, ejes = plt.subplots(2, 5, figsize=(20, 7))

    for eje, variable in zip(ejes.flatten(), VARIABLES_NUMERICAS):
        sns.histplot(data=datos, x=variable, hue=COLUMNA_CLASE, bins=30,
                     palette=["steelblue", "seagreen"], ax=eje, legend=False,
                     stat="density", common_norm=False)
        eje.set_title(variable, fontsize=10)
        eje.set_xlabel("")
        eje.set_ylabel("")

    fig.suptitle("Distribucion de cada variable segun si hubo compra")
    _guardar("histogramas_variables.png")


def graficar_correlacion(datos):
    """Mapa de calor con la correlacion entre variables.
    Identica a la parte 1."""
    plt.figure(figsize=(11, 9))
    sns.heatmap(datos.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    plt.title("Correlacion entre variables")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    _guardar("correlacion.png")


def graficar_matriz_confusion(resultados, titulo, nombre_archivo):
    """
    Mapa de calor de la matriz de confusion.
    Recibe el diccionario que regresa calcular_metricas de metricas.py.
    Identica a la parte 1."""
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


def graficar_curva_n_estimators(resultados):
    """
    Curva de F1 en entrenamiento vs prueba segun n_estimators.
    Equivalente al experimento de profundidad de la parte 1, pero aqui
    la brecha se debe a cuantos arboles se promedian, no a que tan
    profundo es cada uno.
    """
    valores = [r["n_estimators"] for r in resultados]
    f1_entrenamiento = [r["entrenamiento"]["f1"] for r in resultados]
    f1_prueba = [r["prueba"]["f1"] for r in resultados]

    plt.figure(figsize=(9, 5))
    plt.plot(valores, f1_entrenamiento, marker="o",
             color="steelblue", label="Entrenamiento")
    plt.plot(valores, f1_prueba, marker="o",
             color="indianred", label="Prueba")
    plt.fill_between(valores, f1_entrenamiento, f1_prueba,
                     color="gray", alpha=0.15, label="Brecha")
    plt.title("Efecto de n_estimators en el Random Forest")
    plt.xlabel("Numero de arboles (n_estimators)")
    plt.ylabel("F1 Score")
    plt.legend()
    _guardar("curva_n_estimators.png")


def graficar_curva_max_depth(resultados):
    """
    Curva de F1 en entrenamiento vs prueba segun max_depth.
    A diferencia del arbol individual de la parte 1, aqui la brecha
    tarda mas en aparecer porque el bagging reduce el sobreajuste
    de cada arbol individual.
    """
    # None se grafica como un valor mas alla del maximo numerico, para
    # que se vea en el extremo derecho de la curva sin romper el eje
    valores_numericos = [r["max_depth"] for r in resultados if r["max_depth"] is not None]
    tope = max(valores_numericos) + 5

    valores_x = [r["max_depth"] if r["max_depth"] is not None else tope
                for r in resultados]
    etiquetas_x = [str(r["max_depth"]) if r["max_depth"] is not None else "sin\nlimite"
                  for r in resultados]
    f1_entrenamiento = [r["entrenamiento"]["f1"] for r in resultados]
    f1_prueba = [r["prueba"]["f1"] for r in resultados]

    plt.figure(figsize=(9, 5))
    plt.plot(valores_x, f1_entrenamiento, marker="o",
             color="steelblue", label="Entrenamiento")
    plt.plot(valores_x, f1_prueba, marker="o",
             color="indianred", label="Prueba")
    plt.fill_between(valores_x, f1_entrenamiento, f1_prueba,
                     color="gray", alpha=0.15, label="Brecha")
    plt.xticks(valores_x, etiquetas_x)
    plt.title("Efecto de max_depth en el Random Forest")
    plt.xlabel("Profundidad maxima de cada arbol")
    plt.ylabel("F1 Score")
    plt.legend()
    _guardar("curva_max_depth.png")


def graficar_comparacion_modelos(metricas_modelo_1, metricas_modelo_2, metricas_modelo_3):
    """
    Barras comparando los tres modelos (baseline, underfitting a proposito,
    y el optimo) en las cinco metricas, para justificar visualmente
    por que modelo_3 es la mejor configuracion.
    """
    metricas_nombres = ["accuracy", "precision", "recall", "specificity", "f1"]

    etiquetas = metricas_nombres * 3
    valores = ([metricas_modelo_1[m] for m in metricas_nombres] +
               [metricas_modelo_2[m] for m in metricas_nombres] +
               [metricas_modelo_3[m] for m in metricas_nombres])
    modelos = (["modelo_1 (baseline)"] * len(metricas_nombres) +
              ["modelo_2 (underfitting)"] * len(metricas_nombres) +
              ["modelo_3 (optimo)"] * len(metricas_nombres))

    plt.figure(figsize=(11, 5))
    sns.barplot(x=etiquetas, y=valores, hue=modelos,
               palette=["steelblue", "indianred", "seagreen"])
    plt.title("Comparacion de los tres modelos - conjunto de prueba")
    plt.xlabel("")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.legend(title="")
    _guardar("comparacion_modelos.png")


def graficar_importancia_variables(pares_variable_importancia):
    """
    Barras con feature_importances_ del modelo_1, ordenadas de mayor a menor.
    Equivalente a graficar_variables_usadas de la parte 1, pero aqui el
    valor es la importancia ponderada que calcula sklearn, no un conteo
    de nodos.
    """
    nombres = [par[0] for par in pares_variable_importancia]
    valores = [par[1] for par in pares_variable_importancia]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=valores, y=nombres, hue=nombres, palette="viridis", legend=False)
    plt.title("Importancia de las variables (feature_importances_)")
    plt.xlabel("Importancia")
    plt.ylabel("")
    _guardar("importancia_variables.png")


def graficar_arbol_ejemplo(modelo, profundidad_maxima_dibujo=3,
                           nombre_archivo="arbol_ejemplo.png"):
    """
    Dibuja UNO de los arboles del bosque (el primero, modelo.estimators_[0])
    limitado a profundidad_maxima_dibujo niveles, solo para ilustrar como
    luce una de las decisiones individuales dentro del Random Forest.

    Importante para el reporte: esto NO es "el modelo", es solo uno de
    los n_estimators arboles que se promedian. El arbol real que entrena
    sklearn puede ser mucho mas profundo (ver profundidad_promedio en
    modelo.py); aqui solo se recorta la VISUALIZACION a 3 niveles para
    que se pueda leer, el arbol entrenado no se modifica.
    """
    primer_arbol = modelo.estimators_[0]

    plt.figure(figsize=(20, 10))
    plot_tree(primer_arbol,
             max_depth=profundidad_maxima_dibujo,
             feature_names=VARIABLES,
             class_names=["No compro", "Si compro"],
             filled=True,
             rounded=True,
             fontsize=8)
    plt.title(f"Un arbol del Random Forest (arbol #1 de {len(modelo.estimators_)}, "
             f"mostrando los primeros {profundidad_maxima_dibujo} niveles)")
    _guardar(nombre_archivo)