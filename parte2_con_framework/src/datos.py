"""
datos.py
Carga, preparacion y separacion del dataset Online Shoppers Purchasing Intention.

A diferencia de la parte 1, aqui si usamos framework (scikit-learn), asi que los datos los dejo como DataFrame
 de pandas en vez de convertirlos a listas por comodidad personal
"""

import os
import pandas as pd

# Las variables numericas del dataset
VARIABLES_NUMERICAS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

# Variables categoricas que si vamos a usar
VARIABLES_BINARIAS = ["Weekend", "VisitorRecurrente"]

# Todas las variables que entran al modelo
VARIABLES = VARIABLES_NUMERICAS + VARIABLES_BINARIAS

# Nombre que le damos a la clase (la columna original se llama Revenue)
COLUMNA_CLASE = "Compra"

# Ruta al csv relativa
RUTA_DATOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "online_shoppers_intention.csv"
)

def cargar_datos(ruta=RUTA_DATOS):
    #Lee el csv tal cual, con su encabezado original
    return pd.read_csv(ruta)


def preparar_variables(df):
    """
    Deja solo las columnas que entran al modelo y las convierte a numeros.

    - Revenue (TRUE/FALSE) se convierte en Compra (1/0), que es la clase a predecir.
    - Weekend (TRUE/FALSE) se convierte en 1/0.
    - VisitorType se resume en VisitorRecurrente: 1 si es Returning_Visitor, 0 si no.
    - Month, OperatingSystems, Browser, Region y TrafficType se descartan, igual
      que en la parte 1: son categoricos sin orden real y complicarian la
      comparacion directa entre el arbol a mano y el Random Forest de sklearn.
    """
    datos = pd.DataFrame()

    for variable in VARIABLES_NUMERICAS:
        datos[variable] = df[variable].astype(float)

    datos["Weekend"] = df["Weekend"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    datos["VisitorRecurrente"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    datos[COLUMNA_CLASE] = df["Revenue"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

    return datos

def separar_entrenamiento_prueba(datos, proporcion_prueba=0.2, semilla=42):
    """
    Parte en entrenamiento y prueba.
    Misma logica y misma semilla que en la parte 1, para que ambos modelos (arbol a mano y Random Forest) se entrenen y prueben con exactamente la misma particion de datos.
    """
    revueltos = datos.sample(frac=1, random_state=semilla).reset_index(drop=True)

    corte = int(len(revueltos) * (1 - proporcion_prueba))
    entrenamiento = revueltos.iloc[:corte].reset_index(drop=True)
    prueba = revueltos.iloc[corte:].reset_index(drop=True)

    return entrenamiento, prueba

def separar_X_y(datos):
    """
    Separa las variables (X) de la clase a predecir (y).
    A diferencia de la parte 1, aqui se dejan como DataFrame/Series de pandas en vez de convertirlos a listas
    """
    X = datos[VARIABLES]
    y = datos[COLUMNA_CLASE].astype(int)
    return X, y

def distribucion_clases(datos):
    """Cuenta cuantos registros hay de cada clase y su porcentaje."""
    conteos = datos[COLUMNA_CLASE].value_counts().to_dict()
    total = len(datos)
    return {clase: (conteo, 100 * conteo / total) for clase, conteo in conteos.items()}


# Prueba rapida: correr este archivo directamente muestra un resumen del dataset
if __name__ == "__main__":
    df = cargar_datos()
    print("Dimensiones del archivo original:", df.shape)
    print("Columnas:", list(df.columns))
    print()

    datos = preparar_variables(df)
    print("Variables que entran al modelo:", len(VARIABLES))
    print(datos.head())
    print()

    print("Distribucion de clases:")
    for clase, (conteo, porcentaje) in sorted(distribucion_clases(datos).items()):
        etiqueta = "Si compro" if clase == 1 else "No compro"
        print(f"  {clase} ({etiqueta}): {conteo} ({porcentaje:.1f}%)")
    print()

    entrenamiento, prueba = separar_entrenamiento_prueba(datos)
    print("Entrenamiento:", len(entrenamiento), "registros")
    print("Prueba:", len(prueba), "registros")

    X_ent, y_ent = separar_X_y(entrenamiento)
    print("Forma de X entrenamiento:", X_ent.shape)
    print("Ejemplo de registro:\n", X_ent.iloc[0])
    print("Clase de ese registro:", y_ent.iloc[0])