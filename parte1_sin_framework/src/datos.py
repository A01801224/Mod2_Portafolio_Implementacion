"""
datos.py
Carga, preparacion y separacion del dataset Online Shoppers Purchasing Intention.

Se usa pandas solo para leer y manipular la tabla.
El algoritmo es arbol.py
"""

import os
import pandas as pd

# Las 10 variables numericas del dataset
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

# Variables categoricas que si vamos a usar, codificadas a mano como 0 y 1
VARIABLES_BINARIAS = ["Weekend", "VisitorRecurrente"]

# Todas las variables que entran al modelo
VARIABLES = VARIABLES_NUMERICAS + VARIABLES_BINARIAS

# Nombre que le damos a la clase (la columna original se llama Revenue)
COLUMNA_CLASE = "Compra"

# Ruta al csv relativa a este archivo, para que funcione desde cualquier carpeta
RUTA_DATOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "online_shoppers_intention.csv"
)


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el csv tal cual viene, con su encabezado original."""
    return pd.read_csv(ruta)


def preparar_variables(df):
    """
    Deja solo las columnas que entran al modelo y las convierte a numeros.

    - Revenue (TRUE/FALSE) se convierte en Compra (1/0), que es la clase a predecir.
    - Weekend (TRUE/FALSE) se convierte en 1/0.
    - VisitorType se resume en VisitorRecurrente: 1 si es Returning_Visitor, 0 si no.
    - Month, OperatingSystems, Browser, Region y TrafficType se descartan:
      son codigos categoricos sin orden real, y el arbol parte por umbrales
      numericos, asi que un corte del tipo "Browser < 7" no significaria nada.

    La codificacion se hace a mano, sin usar ningun encoder de sklearn.
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
    Revuelve los registros y los parte en entrenamiento y prueba.
    La semilla fija hace que la separacion sea siempre la misma,
    para que los resultados del reporte se puedan reproducir.

    Se hace a mano con sample de pandas, sin usar train_test_split de sklearn.
    """
    revueltos = datos.sample(frac=1, random_state=semilla).reset_index(drop=True)

    corte = int(len(revueltos) * (1 - proporcion_prueba))
    entrenamiento = revueltos.iloc[:corte].reset_index(drop=True)
    prueba = revueltos.iloc[corte:].reset_index(drop=True)

    return entrenamiento, prueba


def a_listas(datos):
    """
    Convierte el DataFrame a listas de Python.
    El arbol trabaja con listas porque esta programado a mano,
    sin depender de las estructuras de pandas.
      X: lista de registros, cada uno es una lista de numeros
      y: lista de etiquetas (0 = no compro, 1 = si compro)
    """
    X = datos[VARIABLES].values.tolist()
    y = datos[COLUMNA_CLASE].astype(int).tolist()
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

    print("Valores faltantes por columna:", int(datos.isnull().sum().sum()))
    print()

    entrenamiento, prueba = separar_entrenamiento_prueba(datos)
    print("Entrenamiento:", len(entrenamiento), "registros")
    print("Prueba:", len(prueba), "registros")

    X_ent, y_ent = a_listas(entrenamiento)
    print("Ejemplo de registro:", [round(valor, 2) for valor in X_ent[0]], "-> clase", y_ent[0])
