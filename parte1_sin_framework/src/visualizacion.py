"""
visualizacion.py
Graficas para el reporte, hechas con seaborn y matplotlib.

Estas bibliotecas solo se usan para visualizar, no para modelar.
Todas las graficas se guardan como png en la carpeta resultados/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from datos import VARIABLES_NUMERICAS, COLUMNA_CLASE

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
    """Barras con cuantas sesiones terminaron en compra. Muestra el desbalance."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=datos, x=COLUMNA_CLASE, hue=COLUMNA_CLASE,
                  palette=["steelblue", "seagreen"], legend=False)
    plt.title("Distribucion de clases")
    plt.xlabel("Compra (0 = No, 1 = Si)")
    plt.ylabel("Cantidad de sesiones")
    _guardar("distribucion_clases.png")


def graficar_histogramas(datos):
    """Un histograma por variable numerica, separado por clase."""
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
    """Mapa de calor con la correlacion entre variables."""
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
    """
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


def graficar_curva_profundidad(resultados):
    """
    Curva de F1 en entrenamiento vs prueba segun la profundidad maxima.
    Es la grafica que muestra el sobreajuste: cuando las dos lineas se
    separan, el arbol dejo de generalizar y empezo a memorizar.
    """
    profundidades = [r["profundidad"] for r in resultados]
    f1_entrenamiento = [r["entrenamiento"]["f1"] for r in resultados]
    f1_prueba = [r["prueba"]["f1"] for r in resultados]

    plt.figure(figsize=(9, 5))
    plt.plot(profundidades, f1_entrenamiento, marker="o",
             color="steelblue", label="Entrenamiento")
    plt.plot(profundidades, f1_prueba, marker="o",
             color="indianred", label="Prueba")
    plt.fill_between(profundidades, f1_entrenamiento, f1_prueba,
                     color="gray", alpha=0.15, label="Brecha (sobreajuste)")
    plt.title("Efecto de la profundidad maxima del arbol")
    plt.xlabel("Profundidad maxima")
    plt.ylabel("F1 Score")
    plt.legend()
    _guardar("curva_profundidad.png")


def graficar_comparacion_criterios(resultados_entropia, resultados_gini):
    """Barras comparando entropia contra Gini en el conjunto de prueba."""
    metricas = ["accuracy", "precision", "recall", "specificity", "f1"]

    etiquetas = metricas * 2
    valores = ([resultados_entropia[m] for m in metricas] +
               [resultados_gini[m] for m in metricas])
    criterios = ["Entropia"] * len(metricas) + ["Gini"] * len(metricas)

    plt.figure(figsize=(9, 5))
    sns.barplot(x=etiquetas, y=valores, hue=criterios,
                palette=["steelblue", "seagreen"])
    plt.title("Entropia vs Gini - conjunto de prueba")
    plt.xlabel("")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.legend(title="")
    _guardar("comparacion_criterios.png")


def graficar_variables_usadas(conteos, nombres_variables):
    """Barras con en cuantos nodos se uso cada variable para partir."""
    pares = [(nombre, conteo) for nombre, conteo
             in zip(nombres_variables, conteos) if conteo > 0]
    pares.sort(key=lambda par: par[1], reverse=True)

    nombres = [par[0] for par in pares]
    valores = [par[1] for par in pares]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=valores, y=nombres, hue=nombres, palette="viridis", legend=False)
    plt.title("Variables mas usadas para partir")
    plt.xlabel("Numero de nodos")
    plt.ylabel("")
    _guardar("variables_usadas.png")


# Correr este archivo directamente genera las graficas exploratorias
if __name__ == "__main__":
    from datos import cargar_datos, preparar_variables

    datos = preparar_variables(cargar_datos())

    graficar_distribucion_clases(datos)
    graficar_histogramas(datos)
    graficar_correlacion(datos)

    print("\nListo. Las graficas exploratorias estan en la carpeta resultados/")
