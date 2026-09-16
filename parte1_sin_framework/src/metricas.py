"""
metricas.py
Matriz de confusion y metricas de evaluacion, calculadas a mano.
No se usa ninguna biblioteca.
"""


def matriz_confusion(y_real, y_predicho):
    """
    Regresa TP, TN, FP, FN tomando la clase 1 (si hubo compra) como la positiva.
      TP: era 1 y se predijo 1
      TN: era 0 y se predijo 0
      FP: era 0 pero se predijo 1  (falsa alarma)
      FN: era 1 pero se predijo 0  (compra que se dejo pasar)
    """
    TP = TN = FP = FN = 0

    for real, predicho in zip(y_real, y_predicho):
        if real == 1 and predicho == 1:
            TP += 1
        elif real == 0 and predicho == 0:
            TN += 1
        elif real == 0 and predicho == 1:
            FP += 1
        elif real == 1 and predicho == 0:
            FN += 1

    return TP, TN, FP, FN


def _division_segura(numerador, denominador):
    """Evita dividir entre cero cuando una clase no aparece en las predicciones."""
    if denominador == 0:
        return 0.0
    return numerador / denominador


def accuracy(TP, TN, FP, FN):
    """Proporcion de aciertos sobre el total."""
    return _division_segura(TP + TN, TP + TN + FP + FN)


def precision(TP, FP):
    """De los que predije como positivos, cuantos si lo eran."""
    return _division_segura(TP, TP + FP)


def recall(TP, FN):
    """De los positivos reales, cuantos alcance a detectar. Tambien llamado sensibilidad."""
    return _division_segura(TP, TP + FN)


def specificity(TN, FP):
    """De los negativos reales, cuantos identifique bien."""
    return _division_segura(TN, TN + FP)


def f1_score(precision_valor, recall_valor):
    """Media armonica entre precision y recall."""
    return _division_segura(2 * precision_valor * recall_valor,
                            precision_valor + recall_valor)


def calcular_metricas(y_real, y_predicho):
    """Calcula todo de un jalon y lo regresa en un diccionario."""
    TP, TN, FP, FN = matriz_confusion(y_real, y_predicho)

    precision_valor = precision(TP, FP)
    recall_valor = recall(TP, FN)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy(TP, TN, FP, FN),
        "precision": precision_valor,
        "recall": recall_valor,
        "specificity": specificity(TN, FP),
        "f1": f1_score(precision_valor, recall_valor)
    }


def imprimir_reporte(resultados, titulo="Resultados"):
    """Imprime la matriz de confusion y las metricas en consola."""
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


# Prueba rapida con datos inventados para verificar que las formulas esten bien
if __name__ == "__main__":
    y_real = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_predicho = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1]

    resultados = calcular_metricas(y_real, y_predicho)
    imprimir_reporte(resultados, "Prueba con datos inventados")
    print("Se esperaba TP=3, FN=1, TN=4, FP=2")
