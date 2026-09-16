"""
metricas.py
Matriz de confusion y metricas de evaluacion, usando sklearn.metrics.

A diferencia de la parte 1, aqui se usa el framework tambien para esta parte, ya que esta permitido y
simplifica el codigo sin perder claridad sobre que se esta midiendo.
"""

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)


def calcular_metricas(y_real, y_predicho):
    """
    Calcula la matriz de confusion y las mismas cinco metricas de la
    parte 1, para poder comparar directamente ambos reportes.
    La clase 1 (si hubo compra) se toma como la clase positiva.
    """
    TN, FP, FN, TP = confusion_matrix(y_real, y_predicho).ravel()

    accuracy = accuracy_score(y_real, y_predicho)
    precision = precision_score(y_real, y_predicho, zero_division=0)
    recall = recall_score(y_real, y_predicho, zero_division=0)
    f1 = f1_score(y_real, y_predicho, zero_division=0)

    # specificity no tiene una funcion directa en sklearn.metrics,
    # asi que se calcula con los valores que ya nos dio confusion_matrix
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }


def imprimir_reporte(resultados, titulo="Resultados"):
    print(titulo)
    print("-" * len(titulo))
    print("Matriz de confusion:")
    print("                  Predicho 0    Predicho 1")
    print(f"  Real 0            {resultados['TN']:>6}        {resultados['FP']:>6}")
    print(f"  Real 1            {resultados['FN']:>6}        {resultados['TP']:>6}")
    print()
    print(f"  Accuracy:     {resultados['accuracy']:.4f}")
    print(f"  Precision:    {resultados['precision']:.4f}")
    print(f"  Recall:       {resultados['recall']:.4f}")
    print(f"  Specificity:  {resultados['specificity']:.4f}")
    print(f"  F1 Score:     {resultados['f1']:.4f}")
    print()