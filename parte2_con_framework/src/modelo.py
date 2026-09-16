"""
modelo.py
Entrenamiento del Random Forest usando scikit-learn.

Aqui es donde se usa el framework: la configuracion, el entrenamiento
y las predicciones las hace RandomForestClassifier. Lo que programamos
nosotros es la seleccion de hiperparametros y los experimentos.
"""
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier

# Configuracion base del modelo, antes de correr los experimentos de
# n_estimators y max_depth. Se usan los valores por defecto de sklearn
# (n_estimators=100, max_depth=None) mas una semilla fija para reproducibilidad.
N_ESTIMATORS_BASE = 100
MAX_DEPTH_BASE = None
SEMILLA = 42

# Configuracion de modelo_2: underfitting a proposito.
# Pocos arboles y muy poco profundos, para que le falte capacidad
# de aprender los patrones del dataset. Sirve como contraejemplo
# en el reporte para justificar por que la configuracion base es mejor.
N_ESTIMATORS_MODELO_2 = 5
MAX_DEPTH_MODELO_2 = 2

# Configuracion de modelo_3: la mas optima.
# Estos valores se definen DESPUES de correr los experimentos de
# n_estimators y max_depth (seccion 2 y 3 de main.py), leyendo el punto
# donde el F1 de prueba es mas alto. Por ahora quedan como None: se
# completan una vez que tengamos esa grafica.
N_ESTIMATORS_MODELO_3 = 100
MAX_DEPTH_MODELO_3 = 10

def entrenar_modelo(X_entrenamiento, y_entrenamiento,
                    n_estimators=N_ESTIMATORS_BASE,
                    max_depth=MAX_DEPTH_BASE):
    """
    Crea y entrena un Random Forest con los hiperparametros dados.
    random_state fijo tanto en el forest como en el bootstrap de cada
    arbol, para que el experimento sea reproducible.
    """
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=SEMILLA
    )
    modelo.fit(X_entrenamiento, y_entrenamiento)
    return modelo


def predecir(modelo, X):
    """Wrapper para que main.py no dependa directamente de la API de sklearn."""
    return modelo.predict(X)


def importancia_variables(modelo, nombres_variables):
    """
    Equivalente a contar_variables_usadas de la parte 1, pero aqui sklearn ya lo calcula por nosotros: feature_importances_ es el promedio de
    cuanto reduce la impureza cada variable a lo largo de todos los arboles del bosque (no es un conteo de nodos, es una medida ponderada).
    """
    importancias = modelo.feature_importances_
    pares = list(zip(nombres_variables, importancias))
    pares.sort(key=lambda par: par[1], reverse=True)
    return pares


def profundidad_promedio(modelo):
    """
    Random Forest no tiene "una" profundidad como el arbol de la parte 1, tiene una por cada arbol individual. Aqui se reporta el promedio y
    el maximo entre todos los arboles del bosque.
    """
    profundidades = [arbol.get_depth() for arbol in modelo.estimators_]
    return sum(profundidades) / len(profundidades), max(profundidades)


def hojas_promedio(modelo):
    """Igual que profundidad_promedio, pero contando hojas por arbol."""
    hojas = [arbol.get_n_leaves() for arbol in modelo.estimators_]
    return sum(hojas) / len(hojas), max(hojas)