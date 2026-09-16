"""
main.py
Corre todo de una:
  1. Resumen del dataset
  2. Experimento 1: efecto de la profundidad maxima (sobreajuste)
  3. Experimento 2: entropia contra Gini
  4. Modelo final con la mejor configuracion
  5. Predicciones de ejemplo en consola
  6. Graficas para el reporte
Se ejecuta con:  python src/main.py
"""

import time

from datos import (cargar_datos, preparar_variables,
                   separar_entrenamiento_prueba, a_listas,
                   distribucion_clases, VARIABLES, COLUMNA_CLASE)
from arbol import (preparar_cortes, construir_arbol, predecir,
                   predecir_registro, contar_nodos, contar_hojas,
                   profundidad_arbol, contar_variables_usadas, imprimir_arbol)
from metricas import calcular_metricas, imprimir_reporte
import visualizacion


# Profundidades a probar en el experimento 1
PROFUNDIDADES = [2, 3, 5, 8, 10, 15, 20, 30]

# Configuracion del modelo final. Se ajusta despues de ver el experimento 1.
PROFUNDIDAD_FINAL = 5
CRITERIO_FINAL = "entropia"
MINIMO_MUESTRAS = 20


def separador(titulo):
    print()
    print(titulo)
    print()


def experimento_profundidad(X_entrenamiento, y_entrenamiento,
                            X_prueba, y_prueba, cortes):
    """
    EXPERIMENTO 1: como afecta la profundidad maxima al desempeno.
    Se entrena un arbol por cada profundidad y se miden las metricas en entrenamiento Y en prueba.
    """
    resultados = []

    for profundidad in PROFUNDIDADES:
        inicio = time.time()

        arbol = construir_arbol(X_entrenamiento, y_entrenamiento, cortes,
                                criterio=CRITERIO_FINAL,
                                profundidad_maxima=profundidad,
                                minimo_muestras=MINIMO_MUESTRAS)

        metricas_entrenamiento = calcular_metricas(
            y_entrenamiento, predecir(arbol, X_entrenamiento))
        metricas_prueba = calcular_metricas(
            y_prueba, predecir(arbol, X_prueba))

        resultados.append({
            "profundidad": profundidad,
            "profundidad_real": profundidad_arbol(arbol),
            "hojas": contar_hojas(arbol),
            "entrenamiento": metricas_entrenamiento,
            "prueba": metricas_prueba,
            "segundos": time.time() - inicio
        })

        print(f"  profundidad {profundidad:>2}: "
              f"{contar_hojas(arbol):>4} hojas | "
              f"F1 entrenamiento {metricas_entrenamiento['f1']:.4f} | "
              f"F1 prueba {metricas_prueba['f1']:.4f} | "
              f"brecha {metricas_entrenamiento['f1'] - metricas_prueba['f1']:+.4f}")

    return resultados


def experimento_criterios(X_entrenamiento, y_entrenamiento,
                          X_prueba, y_prueba, cortes):
    """
    EXPERIMENTO 2: entropia contra Gini.
    Se entrena el mismo arbol, con la misma profundidad y los mismosdatos, cambiando unicamente el criterio de impureza.
    """
    resultados = {}

    for criterio in ("entropia", "gini"):
        inicio = time.time()

        arbol = construir_arbol(X_entrenamiento, y_entrenamiento, cortes,
                                criterio=criterio,
                                profundidad_maxima=PROFUNDIDAD_FINAL,
                                minimo_muestras=MINIMO_MUESTRAS)

        metricas = calcular_metricas(y_prueba, predecir(arbol, X_prueba))

        resultados[criterio] = {
            "arbol": arbol,
            "metricas": metricas,
            "nodos": contar_nodos(arbol),
            "hojas": contar_hojas(arbol),
            "raiz_variable": VARIABLES[arbol.variable],
            "raiz_umbral": arbol.umbral,
            "segundos": time.time() - inicio
        }

        print(f"  {criterio:<10} raiz: {VARIABLES[arbol.variable]} "
              f"< {arbol.umbral:.4f} | {contar_hojas(arbol)} hojas | "
              f"F1 {metricas['f1']:.4f} | {time.time() - inicio:.1f} s")

    return resultados


def predicciones_de_ejemplo(arbol, X_prueba, y_prueba, cuantas=10):
    # correr predicciones en consola.
    
    indices_positivos = [i for i, clase in enumerate(y_prueba) if clase == 1]
    indices_negativos = [i for i, clase in enumerate(y_prueba) if clase == 0]

    mitad = cuantas // 2
    indices = indices_positivos[:mitad] + indices_negativos[:cuantas - mitad]

    print(f"  {'#':>4}  {'PageValues':>12}  {'ProductRel':>11}  "
          f"{'ExitRates':>10}  {'Real':>5}  {'Predicho':>9}  {'Resultado':>10}")

    aciertos = 0
    for i in indices:
        registro = X_prueba[i]
        prediccion = predecir_registro(arbol, registro)
        real = y_prueba[i]
        correcto = prediccion == real
        aciertos += correcto

        print(f"  {i:>4}  {registro[8]:>12.2f}  {registro[4]:>11.0f}  "
              f"{registro[7]:>10.4f}  {real:>5}  {prediccion:>9}  "
              f"{'acierto' if correcto else 'ERROR':>10}")

    print(f"\n  {aciertos} de {len(indices)} correctos en esta muestra")


def main():
    inicio_total = time.time()

    separador("1. DATASET")
    datos = preparar_variables(cargar_datos())
    entrenamiento, prueba = separar_entrenamiento_prueba(datos)

    X_entrenamiento, y_entrenamiento = a_listas(entrenamiento)
    X_prueba, y_prueba = a_listas(prueba)

    print(f"  Registros totales:  {len(datos)}")
    print(f"  Variables usadas:   {len(VARIABLES)}")
    print(f"  Entrenamiento:      {len(X_entrenamiento)} (80%)")
    print(f"  Prueba:             {len(X_prueba)} (20%)")
    print()

    for clase, (conteo, porcentaje) in sorted(distribucion_clases(datos).items()):
        etiqueta = "si compro" if clase == 1 else "no compro"
        print(f"  Clase {clase} ({etiqueta}): {conteo} ({porcentaje:.1f}%)")

    # El piso contra el que hay que comparar todo
    piso = 100 * y_prueba.count(0) / len(y_prueba)
    print(f"\n  Piso de referencia: predecir siempre 0 daria {piso:.1f}% de accuracy")

    cortes = preparar_cortes(X_entrenamiento, 32)

    separador("2. EXPERIMENTO 1 - EFECTO DE LA PROFUNDIDAD")
    resultados_profundidad = experimento_profundidad(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba, cortes)

    mejor = max(resultados_profundidad, key=lambda r: r["prueba"]["f1"])
    print(f"\n  Mejor F1 en prueba: profundidad {mejor['profundidad']} "
          f"con {mejor['prueba']['f1']:.4f}")

    separador("3. EXPERIMENTO 2 - ENTROPIA VS GINI")
    resultados_criterios = experimento_criterios(
        X_entrenamiento, y_entrenamiento, X_prueba, y_prueba, cortes)

    print()
    print(f"  {'Metrica':<14}{'Entropia':>12}{'Gini':>12}{'Diferencia':>12}")
    for metrica in ("accuracy", "precision", "recall", "specificity", "f1"):
        valor_entropia = resultados_criterios["entropia"]["metricas"][metrica]
        valor_gini = resultados_criterios["gini"]["metricas"][metrica]
        print(f"  {metrica:<14}{valor_entropia:>12.4f}{valor_gini:>12.4f}"
              f"{valor_gini - valor_entropia:>+12.4f}")

    separador(f"4. MODELO FINAL - {CRITERIO_FINAL}, profundidad {PROFUNDIDAD_FINAL}")
    arbol_final = construir_arbol(X_entrenamiento, y_entrenamiento, cortes,
                                  criterio=CRITERIO_FINAL,
                                  profundidad_maxima=PROFUNDIDAD_FINAL,
                                  minimo_muestras=MINIMO_MUESTRAS)

    print(f"  Nodos: {contar_nodos(arbol_final)}")
    print(f"  Hojas: {contar_hojas(arbol_final)}")
    print(f"  Profundidad alcanzada: {profundidad_arbol(arbol_final)}")
    print()

    metricas_finales = calcular_metricas(y_prueba, predecir(arbol_final, X_prueba))
    imprimir_reporte(metricas_finales, "Resultados sobre el conjunto de prueba")

    print("  Variables mas usadas para partir:")
    conteos = contar_variables_usadas(arbol_final, len(VARIABLES))
    total_cortes = sum(conteos)
    for conteo, nombre in sorted(zip(conteos, VARIABLES), reverse=True):
        if conteo > 0:
            print(f"    {nombre:<26} {conteo:>4} nodos "
                  f"({100 * conteo / total_cortes:5.1f}%)")

    print()
    print("  Estructura de los primeros 3 niveles:")
    arbol_chico = construir_arbol(X_entrenamiento, y_entrenamiento, cortes,
                                  criterio=CRITERIO_FINAL, profundidad_maxima=3)
    imprimir_arbol(arbol_chico, VARIABLES)

    separador("5. PREDICCIONES DE EJEMPLO")
    predicciones_de_ejemplo(arbol_final, X_prueba, y_prueba)

    separador("6. GRAFICAS")
    visualizacion.graficar_distribucion_clases(datos)
    visualizacion.graficar_histogramas(datos)
    visualizacion.graficar_correlacion(datos)
    visualizacion.graficar_matriz_confusion(
        metricas_finales, "Matriz de confusion - modelo final",
        "matriz_confusion.png")
    visualizacion.graficar_curva_profundidad(resultados_profundidad)
    visualizacion.graficar_comparacion_criterios(
        resultados_criterios["entropia"]["metricas"],
        resultados_criterios["gini"]["metricas"])
    visualizacion.graficar_variables_usadas(conteos, VARIABLES)

    print()
    print(f"Todo listo en {time.time() - inicio_total:.1f} segundos.")


if __name__ == "__main__":
    main()