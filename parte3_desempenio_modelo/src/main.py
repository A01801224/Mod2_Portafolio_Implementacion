"""
main.py
Corre todo el analisis de desempeno de punta a punta:
  1. Resumen del dataset y separacion train/validation/test
  2. Modelo base (antes de regularizar) y su diagnostico (train vs validation)
  3. Busqueda de hiperparametros (GridSearchCV con train/validation)
  4. Modelo regularizado y su diagnostico (train vs validation)
  5. Evaluacion final unica sobre el conjunto de prueba (antes vs despues)
  6. Graficas para el reporte
Se ejecuta con:  python src/main.py
"""

import time
import pandas as pd

from datos import (cargar_datos, preparar_variables, separar_train_val_test,
                   separar_X_y, distribucion_clases, VARIABLES)
from modelo import (entrenar_modelo, predecir, buscar_mejores_hiperparametros,
                    N_ESTIMATORS_BASE, MAX_DEPTH_BASE)
from metricas import calcular_metricas, imprimir_reporte
from diagnostico import diagnosticar, imprimir_diagnostico
import visualizacion


def separador(titulo):
    print()
    print(titulo)
    print()


def main():
    inicio_total = time.time()

    # ---------- 1. DATASET Y SEPARACION ----------
    separador("1. DATASET Y SEPARACION (60% / 20% / 20%)")
    datos = preparar_variables(cargar_datos())
    entrenamiento, validacion, prueba = separar_train_val_test(datos)

    X_entrenamiento, y_entrenamiento = separar_X_y(entrenamiento)
    X_validacion, y_validacion = separar_X_y(validacion)
    X_prueba, y_prueba = separar_X_y(prueba)

    print(f"  Registros totales:  {len(datos)}")
    print(f"  Entrenamiento:      {len(X_entrenamiento)} (60%)")
    print(f"  Validacion:         {len(X_validacion)} (20%)")
    print(f"  Prueba:             {len(X_prueba)} (20%)")
    print()
    for clase, (conteo, porcentaje) in sorted(distribucion_clases(datos).items()):
        etiqueta = "si compro" if clase == 1 else "no compro"
        print(f"  Clase {clase} ({etiqueta}): {conteo} ({porcentaje:.1f}%)")

    # ---------- 2. MODELO BASE Y DIAGNOSTICO "ANTES" ----------
    separador("2. MODELO BASE (antes de regularizar) - DIAGNOSTICO")
    print(f"  Configuracion: n_estimators={N_ESTIMATORS_BASE}, max_depth={MAX_DEPTH_BASE}")
    print("  Entrenado SOLO con el conjunto de entrenamiento.")
    print()

    modelo_base = entrenar_modelo(X_entrenamiento, y_entrenamiento,
                                  n_estimators=N_ESTIMATORS_BASE,
                                  max_depth=MAX_DEPTH_BASE)

    metricas_base_train = calcular_metricas(y_entrenamiento, predecir(modelo_base, X_entrenamiento))
    metricas_base_val = calcular_metricas(y_validacion, predecir(modelo_base, X_validacion))

    diagnostico_antes = diagnosticar(metricas_base_train, metricas_base_val)
    imprimir_diagnostico(diagnostico_antes, "Diagnostico ANTES de regularizar")

    # ---------- 3. BUSQUEDA DE HIPERPARAMETROS ----------
    separador("3. BUSQUEDA DE HIPERPARAMETROS (GridSearchCV con train/validation)")
    inicio = time.time()

    mejores_hiperparametros, mejor_f1_validacion, resultados_grid = buscar_mejores_hiperparametros(
        X_entrenamiento, y_entrenamiento, X_validacion, y_validacion)

    print(f"  Combinaciones probadas: {len(resultados_grid['params'])}")
    print(f"  Tiempo de busqueda:      {time.time() - inicio:.1f} s")
    print(f"  Mejor F1 en validacion:  {mejor_f1_validacion:.4f}")
    print(f"  Mejores hiperparametros: {mejores_hiperparametros}")

    # ---------- 4. MODELO REGULARIZADO Y DIAGNOSTICO "DESPUES" ----------
    separador("4. MODELO REGULARIZADO (despues) - DIAGNOSTICO")
    print(f"  Configuracion: {mejores_hiperparametros}")
    print("  Entrenado SOLO con el conjunto de entrenamiento (misma data que el modelo base,")
    print("  para que la comparacion antes/despues sea justa).")
    print()

    modelo_regularizado = entrenar_modelo(X_entrenamiento, y_entrenamiento, **mejores_hiperparametros)

    metricas_reg_train = calcular_metricas(y_entrenamiento, predecir(modelo_regularizado, X_entrenamiento))
    metricas_reg_val = calcular_metricas(y_validacion, predecir(modelo_regularizado, X_validacion))

    diagnostico_despues = diagnosticar(metricas_reg_train, metricas_reg_val)
    imprimir_diagnostico(diagnostico_despues, "Diagnostico DESPUES de regularizar")

    # ---------- 5. EVALUACION FINAL SOBRE EL CONJUNTO DE PRUEBA ----------
    separador("5. EVALUACION FINAL (conjunto de PRUEBA, se usa una sola vez)")
    print("  Ambos modelos se reentrenan con train+validacion combinados,")
    print("  para aprovechar mas datos en el modelo que se reporta como final,")
    print("  y se evaluan UNA SOLA VEZ contra el conjunto de prueba.")
    print()

    X_train_val = pd.concat([X_entrenamiento, X_validacion], ignore_index=True)
    y_train_val = pd.concat([y_entrenamiento, y_validacion], ignore_index=True)

    modelo_base_final = entrenar_modelo(X_train_val, y_train_val,
                                        n_estimators=N_ESTIMATORS_BASE,
                                        max_depth=MAX_DEPTH_BASE)
    modelo_regularizado_final = entrenar_modelo(X_train_val, y_train_val, **mejores_hiperparametros)

    metricas_finales_antes = calcular_metricas(y_prueba, predecir(modelo_base_final, X_prueba))
    metricas_finales_despues = calcular_metricas(y_prueba, predecir(modelo_regularizado_final, X_prueba))

    imprimir_reporte(metricas_finales_antes, "ANTES de regularizar - conjunto de prueba")
    imprimir_reporte(metricas_finales_despues, "DESPUES de regularizar - conjunto de prueba")

    print("  Mejora en el conjunto de prueba:")
    for metrica in ("accuracy", "precision", "recall", "specificity", "f1"):
        antes = metricas_finales_antes[metrica]
        despues = metricas_finales_despues[metrica]
        print(f"    {metrica:<12} {antes:.4f} -> {despues:.4f}  ({despues - antes:+.4f})")

    # ---------- 6. GRAFICAS ----------
    separador("6. GRAFICAS")
    visualizacion.graficar_diagnostico_antes_despues(diagnostico_antes, diagnostico_despues)
    visualizacion.graficar_resultados_grid_search(resultados_grid)
    visualizacion.graficar_matriz_confusion(
        metricas_finales_antes, "Matriz de confusion - ANTES de regularizar (prueba)",
        "matriz_confusion_antes.png")
    visualizacion.graficar_matriz_confusion(
        metricas_finales_despues, "Matriz de confusion - DESPUES de regularizar (prueba)",
        "matriz_confusion_despues.png")
    visualizacion.graficar_comparacion_final(metricas_finales_antes, metricas_finales_despues)

    print()
    print(f"Todo listo en {time.time() - inicio_total:.1f} segundos.")


if __name__ == "__main__":
    main()

    