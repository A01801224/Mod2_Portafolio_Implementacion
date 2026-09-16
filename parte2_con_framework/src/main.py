"""
main.py
Corre todo de una:
  1. Resumen del dataset
  2. Modelo 1 (baseline, valores por defecto de sklearn)
  3. Experimento 1: efecto de n_estimators
  4. Experimento 2: efecto de max_depth
  5. Modelo 2 (underfitting a proposito) y Modelo 3 (el mas optimo)
  6. Comparacion final de los tres modelos
  7. Predicciones de ejemplo en consola
  8. Graficas para el reporte
Se ejecuta con:  python src/main.py
"""

import time

from datos import (cargar_datos, preparar_variables,
                   separar_entrenamiento_prueba, separar_X_y,
                   distribucion_clases, VARIABLES, COLUMNA_CLASE)
from modelo import (entrenar_modelo, predecir, importancia_variables,
                    profundidad_promedio, hojas_promedio,
                    N_ESTIMATORS_BASE, MAX_DEPTH_BASE,
                    N_ESTIMATORS_MODELO_2, MAX_DEPTH_MODELO_2,
                    N_ESTIMATORS_MODELO_3, MAX_DEPTH_MODELO_3)
from metricas import calcular_metricas, imprimir_reporte
import visualizacion


def separador(titulo):
    print()
    print(titulo)
    print()


# Valores de n_estimators a probar en el experimento 1.
# max_depth se deja fijo (el valor base) para aislar el efecto de n_estimators.
LISTA_N_ESTIMATORS = [5, 10, 25, 50, 100, 150, 200, 300]


def experimento_n_estimators(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """
    EXPERIMENTO 1: como afecta el numero de arboles (n_estimators) al desempeno.
    Se entrena un Random Forest por cada valor y se miden las metricas
    en entrenamiento Y en prueba, igual que el experimento de profundidad
    de la parte 1.
    """
    resultados = []

    for n_estimators in LISTA_N_ESTIMATORS:
        inicio = time.time()

        modelo = entrenar_modelo(X_entrenamiento, y_entrenamiento,
                                 n_estimators=n_estimators,
                                 max_depth=MAX_DEPTH_BASE)

        metricas_entrenamiento = calcular_metricas(
            y_entrenamiento, predecir(modelo, X_entrenamiento))
        metricas_prueba = calcular_metricas(
            y_prueba, predecir(modelo, X_prueba))

        resultados.append({
            "n_estimators": n_estimators,
            "entrenamiento": metricas_entrenamiento,
            "prueba": metricas_prueba,
            "segundos": time.time() - inicio
        })

        print(f"  n_estimators {n_estimators:>3}: "
              f"F1 entrenamiento {metricas_entrenamiento['f1']:.4f} | "
              f"F1 prueba {metricas_prueba['f1']:.4f} | "
              f"brecha {metricas_entrenamiento['f1'] - metricas_prueba['f1']:+.4f} | "
              f"{time.time() - inicio:.2f} s")

    return resultados


# Valores de max_depth a probar en el experimento 2.
# None significa sin limite (los arboles crecen hasta que las hojas
# son puras o hasta min_samples_split/min_samples_leaf, que dejamos
# en su valor por defecto de sklearn).
LISTA_MAX_DEPTH = [2, 3, 5, 8, 10, 15, 20, None]


def experimento_max_depth(X_entrenamiento, y_entrenamiento, X_prueba, y_prueba):
    """
    EXPERIMENTO 2: como afecta la profundidad maxima de cada arbol al
    desempeno del bosque. n_estimators se deja fijo en el valor base
    para aislar el efecto de max_depth.
    """
    resultados = []

    for max_depth in LISTA_MAX_DEPTH:
        inicio = time.time()

        modelo = entrenar_modelo(X_entrenamiento, y_entrenamiento,
                                 n_estimators=N_ESTIMATORS_BASE,
                                 max_depth=max_depth)

        metricas_entrenamiento = calcular_metricas(
            y_entrenamiento, predecir(modelo, X_entrenamiento))
        metricas_prueba = calcular_metricas(
            y_prueba, predecir(modelo, X_prueba))

        resultados.append({
            "max_depth": max_depth,
            "entrenamiento": metricas_entrenamiento,
            "prueba": metricas_prueba,
            "segundos": time.time() - inicio
        })

        etiqueta_profundidad = max_depth if max_depth is not None else "sin limite"
        print(f"  max_depth {str(etiqueta_profundidad):>10}: "
              f"F1 entrenamiento {metricas_entrenamiento['f1']:.4f} | "
              f"F1 prueba {metricas_prueba['f1']:.4f} | "
              f"brecha {metricas_entrenamiento['f1'] - metricas_prueba['f1']:+.4f} | "
              f"{time.time() - inicio:.2f} s")

    return resultados


def predicciones_de_ejemplo(modelo, X_prueba, y_prueba, cuantas=10):
    """Muestra en consola algunas predicciones individuales, mezclando
    ejemplos de ambas clases para no sesgar la muestra hacia la mayoritaria."""
    y_prueba_lista = y_prueba.tolist()
    indices_positivos = [i for i, clase in enumerate(y_prueba_lista) if clase == 1]
    indices_negativos = [i for i, clase in enumerate(y_prueba_lista) if clase == 0]

    mitad = cuantas // 2
    indices = indices_positivos[:mitad] + indices_negativos[:cuantas - mitad]

    predicciones = predecir(modelo, X_prueba)

    print(f"  {'#':>4}  {'PageValues':>12}  {'ProductRel':>11}  "
          f"{'ExitRates':>10}  {'Real':>5}  {'Predicho':>9}  {'Resultado':>10}")

    aciertos = 0
    for i in indices:
        registro = X_prueba.iloc[i]
        prediccion = predicciones[i]
        real = y_prueba_lista[i]
        correcto = prediccion == real
        aciertos += correcto

        print(f"  {i:>4}  {registro['PageValues']:>12.2f}  {registro['ProductRelated']:>11.0f}  "
              f"{registro['ExitRates']:>10.4f}  {real:>5}  {prediccion:>9}  "
              f"{'acierto' if correcto else 'ERROR':>10}")

    print(f"\n  {aciertos} de {len(indices)} correctos en esta muestra")


def main():
    inicio_total = time.time()

    separador("1. DATASET")
    datos = preparar_variables(cargar_datos())
    entrenamiento, prueba = separar_entrenamiento_prueba(datos)

    X_entrenamiento, y_entrenamiento = separar_X_y(entrenamiento)
    X_prueba, y_prueba = separar_X_y(prueba)

    print(f"  Registros totales:  {len(datos)}")
    print(f"  Variables usadas:   {len(VARIABLES)}")
    print(f"  Entrenamiento:      {len(X_entrenamiento)} (80%)")
    print(f"  Prueba:             {len(X_prueba)} (20%)")
    print()

    for clase, (conteo, porcentaje) in sorted(distribucion_clases(datos).items()):
        etiqueta = "si compro" if clase == 1 else "no compro"
        print(f"  Clase {clase} ({etiqueta}): {conteo} ({porcentaje:.1f}%)")

    piso = 100 * (y_prueba == 0).sum() / len(y_prueba)
    print(f"\n  Piso de referencia: predecir siempre 0 daria {piso:.1f}% de accuracy")

    separador("2. MODELO 1 - BASELINE (valores por defecto de sklearn)")
    inicio = time.time()

    modelo_1 = entrenar_modelo(X_entrenamiento, y_entrenamiento,
                               n_estimators=N_ESTIMATORS_BASE,
                               max_depth=MAX_DEPTH_BASE)

    metricas_modelo_1 = calcular_metricas(y_prueba, predecir(modelo_1, X_prueba))

    profundidad_prom, profundidad_max = profundidad_promedio(modelo_1)
    hojas_prom, hojas_max = hojas_promedio(modelo_1)

    print(f"  n_estimators: {N_ESTIMATORS_BASE}  |  max_depth: {MAX_DEPTH_BASE}")
    print(f"  Profundidad promedio de los arboles: {profundidad_prom:.1f} (max {profundidad_max})")
    print(f"  Hojas promedio por arbol:             {hojas_prom:.1f} (max {hojas_max})")
    print(f"  Tiempo de entrenamiento:               {time.time() - inicio:.2f} s")
    print()

    imprimir_reporte(metricas_modelo_1, "Resultados de modelo_1 sobre el conjunto de prueba")

    print("  Variables mas importantes (feature_importances_):")
    importancias_modelo_1 = importancia_variables(modelo_1, VARIABLES)
    for nombre, importancia in importancias_modelo_1[:5]:
        print(f"    {nombre:<26} {importancia:.4f}")

    separador("3. EXPERIMENTO 1 - EFECTO DE N_ESTIMATORS")
    resultados_n_estimators = experimento_n_estimators(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)

    mejor_n = max(resultados_n_estimators, key=lambda r: r["prueba"]["f1"])
    print(f"\n  Mejor F1 en prueba: n_estimators {mejor_n['n_estimators']} "
          f"con {mejor_n['prueba']['f1']:.4f}")

    separador("4. EXPERIMENTO 2 - EFECTO DE MAX_DEPTH")
    resultados_max_depth = experimento_max_depth(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba)

    mejor_profundidad = max(resultados_max_depth, key=lambda r: r["prueba"]["f1"])
    print(f"\n  Mejor F1 en prueba: max_depth {mejor_profundidad['max_depth']} "
          f"con {mejor_profundidad['prueba']['f1']:.4f}")

    separador("5. MODELO 2 (underfitting a proposito) Y MODELO 3 (optimo)")

    modelo_2 = entrenar_modelo(X_entrenamiento, y_entrenamiento,
                               n_estimators=N_ESTIMATORS_MODELO_2,
                               max_depth=MAX_DEPTH_MODELO_2)
    metricas_modelo_2 = calcular_metricas(y_prueba, predecir(modelo_2, X_prueba))

    modelo_3 = entrenar_modelo(X_entrenamiento, y_entrenamiento,
                               n_estimators=N_ESTIMATORS_MODELO_3,
                               max_depth=MAX_DEPTH_MODELO_3)
    metricas_modelo_3 = calcular_metricas(y_prueba, predecir(modelo_3, X_prueba))

    imprimir_reporte(metricas_modelo_2,
                     f"Resultados de modelo_2 (n_estimators={N_ESTIMATORS_MODELO_2}, "
                     f"max_depth={MAX_DEPTH_MODELO_2})")
    imprimir_reporte(metricas_modelo_3,
                     f"Resultados de modelo_3 (n_estimators={N_ESTIMATORS_MODELO_3}, "
                     f"max_depth={MAX_DEPTH_MODELO_3})")

    separador("6. COMPARACION FINAL DE LOS TRES MODELOS")
    print(f"  {'Modelo':<12}{'n_estimators':>14}{'max_depth':>12}{'F1':>10}{'Accuracy':>12}")
    for nombre, n_est, prof, metricas in [
        ("modelo_1", N_ESTIMATORS_BASE, MAX_DEPTH_BASE, metricas_modelo_1),
        ("modelo_2", N_ESTIMATORS_MODELO_2, MAX_DEPTH_MODELO_2, metricas_modelo_2),
        ("modelo_3", N_ESTIMATORS_MODELO_3, MAX_DEPTH_MODELO_3, metricas_modelo_3),
    ]:
        print(f"  {nombre:<12}{str(n_est):>14}{str(prof):>12}"
              f"{metricas['f1']:>10.4f}{metricas['accuracy']:>12.4f}")

    mejor_modelo_nombre = max(
        [("modelo_1", metricas_modelo_1), ("modelo_2", metricas_modelo_2),
         ("modelo_3", metricas_modelo_3)],
        key=lambda par: par[1]["f1"])[0]
    print(f"\n  Mejor de los tres segun F1: {mejor_modelo_nombre}")

    separador("7. PREDICCIONES DE EJEMPLO (con modelo_3)")
    predicciones_de_ejemplo(modelo_3, X_prueba, y_prueba)

    separador("8. GRAFICAS")
    visualizacion.graficar_distribucion_clases(datos)
    visualizacion.graficar_histogramas(datos)
    visualizacion.graficar_correlacion(datos)
    visualizacion.graficar_matriz_confusion(
        metricas_modelo_3, "Matriz de confusion - modelo_3 (optimo)",
        "matriz_confusion.png")
    visualizacion.graficar_curva_n_estimators(resultados_n_estimators)
    visualizacion.graficar_curva_max_depth(resultados_max_depth)
    visualizacion.graficar_comparacion_modelos(
        metricas_modelo_1, metricas_modelo_2, metricas_modelo_3)
    visualizacion.graficar_importancia_variables(importancias_modelo_1)
    visualizacion.graficar_arbol_ejemplo(modelo_3)

    print()
    print(f"Todo listo en {time.time() - inicio_total:.1f} segundos.")


if __name__ == "__main__":
    main()