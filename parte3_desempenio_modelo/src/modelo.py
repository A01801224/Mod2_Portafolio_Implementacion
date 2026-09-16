"""
modelo.py
Entrenamiento del Random Forest y busqueda de hiperparametros.

Reutiliza la misma clase (RandomForestClassifier) y logica base de la
parte 2. Lo nuevo aqui es buscar_mejores_hiperparametros, que usa
GridSearchCV pero con un PredefinedSplit en vez de k-fold: en vez de que
sklearn parta el entrenamiento en varios folds automaticamente, se le dice
explicitamente "entrena con este bloque (train) y evalua con este otro
(validation)", para que la busqueda respete exactamente la separacion de
tres vias que pide este entregable, en vez de mezclar datos de validacion
con el entrenamiento a traves de un k-fold interno.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit

SEMILLA = 42

# Configuracion base (identica a modelo_1 de la parte 2), que aqui se usa
# como el modelo a diagnosticar ANTES de regularizar.
N_ESTIMATORS_BASE = 100
MAX_DEPTH_BASE = None

# Grid de hiperparametros para la busqueda. Es mas amplio que el de la
# parte 2: ademas de n_estimators y max_depth, incluye min_samples_leaf
# (tecnica de regularizacion: obliga a que cada hoja tenga un minimo de
# muestras, evitando hojas hiperespecificas de 1 o 2 registros) y
# class_weight (para compensar el desbalance de clases directamente en
# el entrenamiento, en vez de solo ajustar la profundidad). Revisado Profesor Israel 
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 8, 10, 15, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "class_weight": [None, "balanced"]
}


def entrenar_modelo(X_entrenamiento, y_entrenamiento, **hiperparametros):
    """
    Crea y entrena un Random Forest con los hiperparametros dados.
    Los hiperparametros no especificados usan los valores por defecto
    de sklearn. random_state fijo para reproducibilidad.
    """
    modelo = RandomForestClassifier(random_state=SEMILLA, **hiperparametros)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    return modelo


def predecir(modelo, X):
    """Wrapper simple sobre modelo.predict."""
    return modelo.predict(X)


def buscar_mejores_hiperparametros(X_entrenamiento, y_entrenamiento,
                                   X_validacion, y_validacion,
                                   param_grid=PARAM_GRID):
    """
    Busca la mejor combinacion de hiperparametros dentro de param_grid,
    usando F1 sobre el conjunto de VALIDACION como criterio (nunca el de
    prueba). Regresa los mejores hiperparametros encontrados, el mejor F1
    de validacion, y la tabla completa de resultados (para graficar despues).

    Tecnicamente combina X_entrenamiento y X_validacion en un solo arreglo
    y usa PredefinedSplit para decirle a GridSearchCV exactamente cuales
    filas son de entrenamiento (-1, se excluyen de la evaluacion) y
    cuales son de validacion (0, se usan para evaluar cada combinacion).
    Esto es equivalente a "un solo fold" hecho a la medida de nuestra
    separacion, en vez del k-fold que GridSearchCV usa por defecto.
    """
    X_combinado = np.vstack([X_entrenamiento.values, X_validacion.values])
    y_combinado = np.concatenate([y_entrenamiento.values, y_validacion.values])

    # -1 = fila de entrenamiento (nunca se evalua), 0 = fila de validacion
    test_fold = np.concatenate([
        np.full(len(X_entrenamiento), -1),
        np.zeros(len(X_validacion))
    ])
    particion = PredefinedSplit(test_fold)

    busqueda = GridSearchCV(
        estimator=RandomForestClassifier(random_state=SEMILLA),
        param_grid=param_grid,
        cv=particion,
        scoring="f1",
        n_jobs=-1,
        refit=False  # no reentrena automaticamente: lo hacemos nosotros,
                     # a proposito, sobre train solamente primero (para
                     # el diagnostico "despues") y luego sobre train+val
                     # (para la evaluacion final)
    )
    busqueda.fit(X_combinado, y_combinado)

    return busqueda.best_params_, busqueda.best_score_, busqueda.cv_results_

#l grid tiene 3×5×4×2 = 120 combinaciones, así que 
#al correrlo más adelante puede tardar uno o dos minutos