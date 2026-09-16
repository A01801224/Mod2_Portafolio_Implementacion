"""
arbol.py
Arbol de decision desde cero
"""

import math


def entropia(conteo_positivos, conteo_negativos):
    """
    Criterio de impureza basado en teoria de la informacion.
    Formula a utilizar basado en diapositivas:  -p+ log2(p+) - p- log2(p-)

    Va de 0 a 1 cuando hay dos clases:
      0 = todos de la misma clase
      1 = grupo revuelto, mitad y mitad
    """
    total = conteo_positivos + conteo_negativos

    if total == 0:
        return 0.0

    resultado = 0.0
    for conteo in (conteo_positivos, conteo_negativos):
        # El if implementa  0 log2(0) = 0
        if conteo > 0:
            proporcion = conteo / total
            resultado -= proporcion * math.log2(proporcion)

    return resultado


def gini(conteo_positivos, conteo_negativos):
    """
    Formula a seguir:  1 - (p+)^2 - (p-)^2

    Va de 0 a 0.5 cuando hay dos clases:
      0   = grupo puro
      0.5 = mitad y mitad

    No tinene la misma escals entonces no se pueden comparar los valores de gini con lo de entropia
    """
    total = conteo_positivos + conteo_negativos

    if total == 0:
        return 0.0

    proporcion_positivos = conteo_positivos / total
    proporcion_negativos = conteo_negativos / total

    return 1.0 - (proporcion_positivos ** 2 + proporcion_negativos ** 2)


def impureza(conteo_positivos, conteo_negativos, criterio="entropia"):
    """
    Sirve para que el resto del arbol no sepa cual criterio se usa, solo llama a impuersa, ára que cundo se hagan las dos corridas, solo se cambie este parametro en un solo lugar en ves de mover el algoritmo
    recibe el criterio a usar 
    """
    if criterio == "entropia":
        return entropia(conteo_positivos, conteo_negativos)
    elif criterio == "gini":
        return gini(conteo_positivos, conteo_negativos)
    else:
        raise ValueError(f"Criterio desconocido: {criterio}")


def ganancia(grupos, criterio="entropia"):
    """
    Cuanta impureza se elimina al partir un grupo en varios subgrupos.
    grupos: lista de tuplas (positivos, negativos), una por cada subgrupo resultante del corte.
    Regresa un numero: entre mas grande, mejor el corte.
    """
    total_positivos = sum(positivos for positivos, _ in grupos)
    total_negativos = sum(negativos for _, negativos in grupos)
    total = total_positivos + total_negativos

    if total == 0:
        return 0.0

    # Lo que estaba ANTES de partir
    impureza_antes = impureza(total_positivos, total_negativos, criterio)

    # Lo  que quedo DESPUES, pesando cada grupo por su tamaño
    impureza_despues = 0.0
    for positivos, negativos in grupos:
        tamano_grupo = positivos + negativos
        peso = tamano_grupo / total
        impureza_despues += peso * impureza(positivos, negativos, criterio)

    return impureza_antes - impureza_despues


def cortes_candidatos(valores, numero_cortes=32):
    """
    En vez de probar todos los valores posibles como umbral, se toman unos cuantos repartidos parejo a lo largo del rango. Con 32 cortes,
    cada uno cae aproximadamente cada 3% de los datos.

    Se quitan los repetidos: si una variable tiene pocos valores distintos (por ejemplo Weekend, que solo vale 0 o 1), muchos
    percentiles caen en el mismo numero.
    """
    valores_ordenados = sorted(valores)
    candidatos = []

    for k in range(1, numero_cortes):
        posicion = int(len(valores_ordenados) * k / numero_cortes)
        candidatos.append(valores_ordenados[posicion])

    return sorted(set(candidatos))


def contar_grupos(columna, y, umbral):
    """
    Aplica un corte y cuenta como quedaron los dos grupos.

    columna: los valores de UNA variable, uno por registro, y: las clases, 1 = positivo, 0 = negativo umbral: el numero contra el que se compara

    Regla: si el valor es MENOR que el umbral va a la izquierda, si no, a la derecha.

    Regresa el formato que espera ganancia(): [(positivos_izq, negativos_izq), (positivos_der, negativos_der)]
    """
    positivos_izquierda = negativos_izquierda = 0
    positivos_derecha = negativos_derecha = 0

    for valor, clase in zip(columna, y):
        if valor < umbral:
            if clase == 1:
                positivos_izquierda += 1
            else:
                negativos_izquierda += 1
        else:
            if clase == 1:
                positivos_derecha += 1
            else:
                negativos_derecha += 1

    return [(positivos_izquierda, negativos_izquierda),
            (positivos_derecha, negativos_derecha)]


def preparar_cortes(X, numero_cortes=32):
    """
    Calcula los cortes candidatos de TODAS las variables, una sola vez.

    Esto se hace al principio del entrenamiento, no en cada nodo. 
    Regresa una lista con los umbrales de cada variable.
    """
    numero_variables = len(X[0])

    return [cortes_candidatos([registro[j] for registro in X], numero_cortes)
            for j in range(numero_variables)]


def mejor_corte(X, y, cortes, criterio="entropia", variables_candidatas=None):
    """
    Prueba todos los cortes candidatos y regresa el que mas ganancia da.

    X:lista de registros, y:lista de clases
    cortes: lo que regresa preparar_cortes()
    variables_candidatas: cuales variables considerar.
            None = todas.

    Regresa (variable, umbral, ganancia).
    Si ningun corte mejora nada, regresa (None, None, 0.0).
    """
    if variables_candidatas is None:
        variables_candidatas = range(len(X[0]))

    mejor_variable = None
    mejor_umbral = None
    mejor_ganancia = 0.0

    for variable in variables_candidatas:
        # Se extrae la columna una vez por variable, no una vez por umbral
        columna = [registro[variable] for registro in X]

        for umbral in cortes[variable]:
            grupos = contar_grupos(columna, y, umbral)
            ganancia_corte = ganancia(grupos, criterio)

            if ganancia_corte > mejor_ganancia:
                mejor_ganancia = ganancia_corte
                mejor_variable = variable
                mejor_umbral = umbral

    return mejor_variable, mejor_umbral, mejor_ganancia


class Nodo:
    """
    Una pieza del arbol. Puede ser de dos tipos:
- Nodo interno: hace una pregunta ("PageValues < 0.58?") y tiene dos hijos, uno para cada respuesta.
    - Hoja: ya no pregunta nada, solo dice que clase predice.
    Se distingue uno de otro con es_hoja().
    """

    def __init__(self, variable=None, umbral=None, izquierda=None,
                 derecha=None, clase=None, positivos=0, negativos=0):
        # Solo en nodos internos
        self.variable = variable      # indice de la variable por la que parte
        self.umbral = umbral          # valor del corte
        self.izquierda = izquierda    # subarbol de los que cumplen valor < umbral
        self.derecha = derecha        # subarbol de los demas

        # Solo en hojas
        self.clase = clase            # 0 o 1, lo que predice

        # En ambos, sirve para inspeccionar el arbol despues
        self.positivos = positivos
        self.negativos = negativos

    def es_hoja(self):
        return self.clase is not None


def construir_arbol(X, y, cortes, criterio="entropia", profundidad_maxima=10,
                    minimo_muestras=20, profundidad=0):
    """
    Construye el arbol recursivamente.
    """
    positivos = sum(y)
    negativos = len(y) - positivos

    # --- CRITERIOS DE PARO ---
    # 1. Nodo puro: todos los registros son de la misma clase.
    #    Es el caso base del ID3, ya no hay nada que preguntar.
    # 2. Muy pocos registros: partir grupos minusculos memoriza ruido
    #    en vez de aprender patrones. Es la principal defensa contra
    #    el sobreajuste que menciona la lamina 22.
    # 3. Demasiado profundo: limita el tamano del arbol.
    if (positivos == 0 or negativos == 0
            or len(y) < minimo_muestras
            or profundidad >= profundidad_maxima):
        return crear_hoja(positivos, negativos)

    variable, umbral, ganancia_corte = mejor_corte(X, y, cortes, criterio)

    # 4. Ningun corte mejora nada: tambien se vuelve hoja.
    if variable is None:
        return crear_hoja(positivos, negativos)

    # --- PARTIR LOS DATOS ---
    X_izquierda, y_izquierda = [], []
    X_derecha, y_derecha = [], []

    for registro, clase in zip(X, y):
        if registro[variable] < umbral:
            X_izquierda.append(registro)
            y_izquierda.append(clase)
        else:
            X_derecha.append(registro)
            y_derecha.append(clase)

    # Proteccion: si un lado quedo vacio no tiene caso partir
    if len(y_izquierda) == 0 or len(y_derecha) == 0:
        return crear_hoja(positivos, negativos)

    # --- LA RECURSION ---
    # Cada mitad es el mismo problema, mas chico. La funcion se llama
    # a si misma con profundidad + 1 hasta topar con un criterio de paro.
    return Nodo(
        variable=variable,
        umbral=umbral,
        izquierda=construir_arbol(X_izquierda, y_izquierda, cortes, criterio,
                                  profundidad_maxima, minimo_muestras,
                                  profundidad + 1),
        derecha=construir_arbol(X_derecha, y_derecha, cortes, criterio,
                                profundidad_maxima, minimo_muestras,
                                profundidad + 1),
        positivos=positivos,
        negativos=negativos
    )


def crear_hoja(positivos, negativos):
    """
    Crea una hoja. Predice la clase mayoritaria de los registros que llegaron hasta ahi. En empate predice 0, que es la clase
    mas comun en el dataset.
    """
    clase = 1 if positivos > negativos else 0
    return Nodo(clase=clase, positivos=positivos, negativos=negativos)


def imprimir_arbol(nodo, nombres_variables, profundidad=0, prefijo="raiz"):
    """Dibuja el arbol en la consola para poder inspeccionarlo."""
    sangria = "  " * profundidad

    if nodo.es_hoja():
        total = nodo.positivos + nodo.negativos
        print(f"{sangria}{prefijo}: predice {nodo.clase} "
              f"({nodo.positivos} compraron, {nodo.negativos} no, {total} total)")
        return

    print(f"{sangria}{prefijo}: {nombres_variables[nodo.variable]} < {nodo.umbral:.4f}")
    imprimir_arbol(nodo.izquierda, nombres_variables, profundidad + 1, "si ->")
    imprimir_arbol(nodo.derecha, nombres_variables, profundidad + 1, "no ->")


def predecir_registro(nodo, registro):
    """
    Recorre el arbol desde la raiz hasta una hoja y regresa la clase.

    En cada nodo interno se compara el valor del registro contra elumbral y se baja por la rama que corresponde. Se repite hasta
    llegar a una hoja, que es la que dice la prediccion.
    """
    while not nodo.es_hoja():
        if registro[nodo.variable] < nodo.umbral:
            nodo = nodo.izquierda
        else:
            nodo = nodo.derecha

    return nodo.clase


def predecir(nodo, X):
    """Predice una lista completa de registros."""
    return [predecir_registro(nodo, registro) for registro in X]


def contar_nodos(nodo):
    """Cuantos nodos tiene el arbol en total."""
    if nodo.es_hoja():
        return 1
    return 1 + contar_nodos(nodo.izquierda) + contar_nodos(nodo.derecha)


def profundidad_arbol(nodo):
    """Que tan profundo crecio realmente el arbol."""
    if nodo.es_hoja():
        return 0
    return 1 + max(profundidad_arbol(nodo.izquierda),
                   profundidad_arbol(nodo.derecha))


def contar_hojas(nodo):
    """Cuantas hojas tiene el arbol. Es una medida del tamano del modelo."""
    if nodo.es_hoja():
        return 1
    return contar_hojas(nodo.izquierda) + contar_hojas(nodo.derecha)


def contar_variables_usadas(nodo, numero_variables):
    """
    Cuenta en cuantos nodos se uso cada variable para partir. Si una variable aparece en muchos nodos es porque resulta util para separar las clases.
    """
    conteos = [0] * numero_variables

    def recorrer(nodo_actual):
        if nodo_actual.es_hoja():
            return
        conteos[nodo_actual.variable] += 1
        recorrer(nodo_actual.izquierda)
        recorrer(nodo_actual.derecha)

    recorrer(nodo)
    return conteos


# Pruebas: correr este archivo directamente verifica las formulas
if __name__ == "__main__":
    import time
    from datos import (cargar_datos, preparar_variables,
                       separar_entrenamiento_prueba, a_listas, VARIABLES)
    from metricas import calcular_metricas, imprimir_reporte

    print("ENTROPIA")
    print("  (9, 5) =", round(entropia(9, 5), 4), "-> se espera 0.9403 (Play Tennis)")
    print("  (4, 0) =", round(entropia(4, 0), 4), "-> se espera 0.0 (grupo puro)")
    print("  (7, 7) =", round(entropia(7, 7), 4), "-> se espera 1.0 (maxima impureza)")
    print()

    print("GINI")
    print("  (9, 5) =", round(gini(9, 5), 4), "-> se espera 0.4592")
    print("  (4, 0) =", round(gini(4, 0), 4), "-> se espera 0.0 (grupo puro)")
    print("  (7, 7) =", round(gini(7, 7), 4), "-> se espera 0.5 (maxima impureza)")
    print()

    print("GANANCIA - ejemplo Play Tennis")
    print("  Outlook     =", round(ganancia([(2, 3), (4, 0), (3, 2)]), 4), "-> se espera 0.2467")
    print("  Humidity    =", round(ganancia([(3, 4), (6, 1)]), 4), "-> se espera 0.1518")
    print("  Wind        =", round(ganancia([(6, 2), (3, 3)]), 4), "-> se espera 0.0481")
    print("  Temperature =", round(ganancia([(2, 2), (4, 2), (3, 1)]), 4), "-> se espera 0.0292")
    print()

    print("CORTES Y GRUPOS")
    valores = [1, 2, 3, 4, 5, 6, 7, 8]
    clases = [0, 0, 0, 0, 1, 1, 1, 1]
    print("  cortes_candidatos =", cortes_candidatos(valores, 4), "-> se espera [3, 5, 7]")
    for umbral in (5, 3, 7):
        grupos = contar_grupos(valores, clases, umbral)
        print(f"  umbral {umbral} -> {grupos} ganancia = {round(ganancia(grupos), 4)}")
    print()

    print("MEJOR CORTE - ejemplo controlado")
    X_prueba_chico = [[1, 50], [2, 10], [3, 90], [4, 30],
                      [5, 70], [6, 20], [7, 60], [8, 40]]
    y_prueba_chico = [0, 0, 0, 0, 1, 1, 1, 1]
    cortes_chicos = preparar_cortes(X_prueba_chico, 4)
    print("  mejor corte:", mejor_corte(X_prueba_chico, y_prueba_chico, cortes_chicos),
          "-> se espera (0, 5, 1.0)")
    print()

    # A partir de aqui se trabaja con los datos reales
    entrenamiento, prueba = separar_entrenamiento_prueba(
        preparar_variables(cargar_datos()))
    X_entrenamiento, y_entrenamiento = a_listas(entrenamiento)
    X_prueba, y_prueba = a_listas(prueba)

    cortes = preparar_cortes(X_entrenamiento, 32)

    print("DATOS REALES")
    variable, umbral, ganancia_raiz = mejor_corte(
        X_entrenamiento, y_entrenamiento, cortes)
    print(f"  Raiz: {VARIABLES[variable]} < {umbral:.4f}, ganancia {ganancia_raiz:.4f}")
    print()

    print("ARBOL - primeros 3 niveles")
    arbol_chico = construir_arbol(X_entrenamiento, y_entrenamiento, cortes,
                                  criterio="entropia", profundidad_maxima=3)
    imprimir_arbol(arbol_chico, VARIABLES)
    print()

    print("ARBOL COMPLETO - profundidad 10")
    inicio = time.time()
    arbol_completo = construir_arbol(X_entrenamiento, y_entrenamiento, cortes,
                                     criterio="entropia",
                                     profundidad_maxima=10,
                                     minimo_muestras=20)
    print(f"  Entrenado en {time.time() - inicio:.1f} segundos")
    print(f"  Nodos: {contar_nodos(arbol_completo)}")
    print(f"  Hojas: {contar_hojas(arbol_completo)}")
    print(f"  Profundidad alcanzada: {profundidad_arbol(arbol_completo)}")
    print()

    imprimir_reporte(
        calcular_metricas(y_prueba, predecir(arbol_completo, X_prueba)),
        "Arbol solo - conjunto de prueba")
