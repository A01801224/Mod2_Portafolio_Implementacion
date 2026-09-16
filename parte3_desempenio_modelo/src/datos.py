"""
datos.py
Carga, preparacion y separacion del dataset Online Shoppers Purchasing Intention.

A diferencia de la parte 2, aqui la separacion es de tres vias
(train / validation / test) en vez de dos, porque el analisis de
bias/varianza necesita un conjunto de validacion separado del de prueba:
el de prueba se reserva para la evaluacion final, y el de validacion es
el que se usa para diagnosticar y para elegir hiperparametros.
"""

import os
import pandas as pd

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
VARIABLES_BINARIAS = ["Weekend", "VisitorRecurrente"]
VARIABLES = VARIABLES_NUMERICAS + VARIABLES_BINARIAS
COLUMNA_CLASE = "Compra"

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
    Identica a la parte 2: deja solo las 12 variables que entran al modelo
    y codifica a mano Weekend, VisitorRecurrente y Compra (Revenue).
    """
    datos = pd.DataFrame()

    for variable in VARIABLES_NUMERICAS:
        datos[variable] = df[variable].astype(float)

    datos["Weekend"] = df["Weekend"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    datos["VisitorRecurrente"] = (df["VisitorType"] == "Returning_Visitor").astype(int)
    datos[COLUMNA_CLASE] = df["Revenue"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})

    return datos


def distribucion_clases(datos):
    """Cuenta cuantos registros hay de cada clase y su porcentaje."""
    conteos = datos[COLUMNA_CLASE].value_counts().to_dict()
    total = len(datos)
    return {clase: (conteo, 100 * conteo / total) for clase, conteo in conteos.items()}


def separar_X_y(datos):
    """Separa las variables (X) de la clase a predecir (y), como DataFrame/Series."""
    X = datos[VARIABLES]
    y = datos[COLUMNA_CLASE].astype(int)
    return X, y

def separar_train_val_test(datos, proporcion_train=0.6, proporcion_validation=0.2, semilla=42):
    """
    Revuelve los registros y los parte en tres conjuntos: entrenamiento,
    validacion y prueba. Misma logica que separar_entrenamiento_prueba de
    la parte 2 (sample con semilla fija), extendida a un corte mas.

    - entrenamiento: se usa para ajustar los pesos/arboles del modelo.
    - validacion: se usa para diagnosticar bias/varianza y para elegir
      hiperparametros en el GridSearchCV. El modelo nunca se entrena con
      estos datos, pero se usan muchas veces para tomar decisiones.
    - prueba: se reserva y NO se toca hasta la evaluacion final, para que
      el numero reportado al final sea una estimacion honesta y no este
      contaminado por haberse usado para elegir nada.

    La proporcion de prueba se calcula como el restante
    (1 - proporcion_train - proporcion_validation), que con los valores
    por defecto da 60% / 20% / 20%.
    """
    revueltos = datos.sample(frac=1, random_state=semilla).reset_index(drop=True)

    n = len(revueltos)
    corte_train = int(n * proporcion_train)
    corte_validation = int(n * (proporcion_train + proporcion_validation))

    entrenamiento = revueltos.iloc[:corte_train].reset_index(drop=True)
    validacion = revueltos.iloc[corte_train:corte_validation].reset_index(drop=True)
    prueba = revueltos.iloc[corte_validation:].reset_index(drop=True)

    return entrenamiento, validacion, prueba

# Prueba rapida: correr este archivo muestra un resumen de la separacion
if __name__ == "__main__":
    df = cargar_datos()
    datos = preparar_variables(df)

    entrenamiento, validacion, prueba = separar_train_val_test(datos)

    print("Registros totales:", len(datos))
    print(f"Entrenamiento: {len(entrenamiento)} ({100 * len(entrenamiento) / len(datos):.1f}%)")
    print(f"Validacion:    {len(validacion)} ({100 * len(validacion) / len(datos):.1f}%)")
    print(f"Prueba:        {len(prueba)} ({100 * len(prueba) / len(datos):.1f}%)")
    print()

    for nombre, subconjunto in [("Entrenamiento", entrenamiento),
                                ("Validacion", validacion),
                                ("Prueba", prueba)]:
        print(f"Distribucion de clases en {nombre}:")
        for clase, (conteo, porcentaje) in sorted(distribucion_clases(subconjunto).items()):
            etiqueta = "si compro" if clase == 1 else "no compro"
            print(f"  {clase} ({etiqueta}): {conteo} ({porcentaje:.1f}%)")
        print()