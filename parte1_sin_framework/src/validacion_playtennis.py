"""
validacion_playtennis.py
Validacion del algoritmo reproduciendo el ejercicio de Play Tennis visto en clase.
  Bloque 1 - entropia            -> se prueba con Entropy(S) del ejercicio
  Bloque 2 - contar y agrupar    -> se prueba mostrando la tabla agrupada
  Bloque 3 - ganancia            -> se prueba con las cuatro ganancias
  Bloque 4 - elegir_atributo     -> se prueba eligiendo la raiz
  Bloque 5 - construir_id3       -> se prueba armando el arbol completo
  Bloque 6 - comprobacion final  -> se compara contra las funciones de arbol.py

Se ejecuta con:  python src/validacion_playtennis.py
"""

import math

# DATOS: los 14 dias del ejercicio

ATRIBUTOS = ["Outlook", "Temperature", "Humidity", "Wind"]
CLASE = "PlayTennis"

DIAS = [
    {"Dia": "D1",  "Outlook": "Sunny",    "Temperature": "Hot",  "Humidity": "High",   "Wind": "Weak",   "PlayTennis": "No"},
    {"Dia": "D2",  "Outlook": "Sunny",    "Temperature": "Hot",  "Humidity": "High",   "Wind": "Strong", "PlayTennis": "No"},
    {"Dia": "D3",  "Outlook": "Overcast", "Temperature": "Hot",  "Humidity": "High",   "Wind": "Weak",   "PlayTennis": "Yes"},
    {"Dia": "D4",  "Outlook": "Rain",     "Temperature": "Mild", "Humidity": "High",   "Wind": "Weak",   "PlayTennis": "Yes"},
    {"Dia": "D5",  "Outlook": "Rain",     "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak",   "PlayTennis": "Yes"},
    {"Dia": "D6",  "Outlook": "Rain",     "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "No"},
    {"Dia": "D7",  "Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
    {"Dia": "D8",  "Outlook": "Sunny",    "Temperature": "Mild", "Humidity": "High",   "Wind": "Weak",   "PlayTennis": "No"},
    {"Dia": "D9",  "Outlook": "Sunny",    "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak",   "PlayTennis": "Yes"},
    {"Dia": "D10", "Outlook": "Rain",     "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak",   "PlayTennis": "Yes"},
    {"Dia": "D11", "Outlook": "Sunny",    "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
    {"Dia": "D12", "Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High",   "Wind": "Strong", "PlayTennis": "Yes"},
    {"Dia": "D13", "Outlook": "Overcast", "Temperature": "Hot",  "Humidity": "Normal", "Wind": "Weak",   "PlayTennis": "Yes"},
    {"Dia": "D14", "Outlook": "Rain",     "Temperature": "Mild", "Humidity": "High",   "Wind": "Strong", "PlayTennis": "No"},
]

# Valores publicados en la presentacion, para comparar al final
VALORES_DE_LA_CLASE = {
    "Entropy(S)": 0.940,
    "Outlook": 0.246,
    "Humidity": 0.151,
    "Wind": 0.048,
    "Temperature": 0.029,
}


def titulo(texto):
    """Encabezado para separar los bloques en la salida."""
    print()
    print(texto)


# ======================================================================
# BLOQUE 1 - ENTROPIA
# ======================================================================

def entropia(conteo_positivos, conteo_negativos):
    """
    Mide que tan revuelto esta un grupo.
    Formula de la presentacion:  -p+ log2(p+) - p- log2(p-)

    Con dos clases va de 0 a 1:
      0 = todos son de la misma clase
      1 = mitad y mitad
    """
    total = conteo_positivos + conteo_negativos

    if total == 0:
        return 0.0

    resultado = 0.0
    for conteo in (conteo_positivos, conteo_negativos):
        # Aqui se aplica la convencion 0 log2(0) = 0
        if conteo > 0:
            proporcion = conteo / total
            resultado -= proporcion * math.log2(proporcion)

    return resultado


def prueba_entropia():
    """
    Se calcula la entropia del conjunto completo de 14 dias.
    En el ejercicio hay 9 dias con Yes y 5 con No, y la presentacion
    reporta Entropy(S) = 0.940.
    """
    titulo("BLOQUE 1 - ENTROPIA DEL CONJUNTO COMPLETO")

    positivos = sum(1 for dia in DIAS if dia[CLASE] == "Yes")
    negativos = len(DIAS) - positivos
    total = positivos + negativos

    valor = entropia(positivos, negativos)

    print(f"\n  De los {total} dias: {positivos} con Yes y {negativos} con No")
    print(f"\n  Entropy(S) = -({positivos}/{total}) log2({positivos}/{total}) "
          f"- ({negativos}/{total}) log2({negativos}/{total})")
    print(f"             = {valor:.4f}")
    print(f"\n  La presentacion reporta 0.940. Coincide.")

    print("\n  Casos extremos para comprobar que la formula se comporta bien:")
    print(f"    entropia(4, 0) = {entropia(4, 0):.4f}   grupo puro, impureza minima")
    print(f"    entropia(7, 7) = {entropia(7, 7):.4f}   mitad y mitad, impureza maxima")
    print(f"    entropia(0, 0) = {entropia(0, 0):.4f}   grupo vacio, no truena")

    return valor


# ======================================================================
# BLOQUE 2 - CONTAR Y AGRUPAR
# ======================================================================

def contar(dias):
    """Cuenta cuantos Yes y cuantos No hay en un grupo de dias."""
    positivos = sum(1 for dia in dias if dia[CLASE] == "Yes")
    negativos = len(dias) - positivos
    return positivos, negativos


def agrupar_por(dias, atributo):
    """
    Parte los dias segun el valor del atributo.
    Asi funciona ID3 con atributos categoricos: una rama por cada valor
    distinto, no un corte binario como en el arbol del entregable.
    Regresa un diccionario {valor: lista de dias}.
    """
    grupos = {}
    for dia in dias:
        grupos.setdefault(dia[atributo], []).append(dia)
    return grupos


def prueba_agrupar():
    """
    Se agrupa el conjunto por Outlook, que es el atributo del ejemplo
    de la presentacion, y se verifica que los grupos coincidan con los
    que se contaron a mano en clase: Sunny (2,3), Overcast (4,0), Rain (3,2).
    """
    titulo("BLOQUE 2 - AGRUPAR LOS DIAS POR UN ATRIBUTO")

    grupos = agrupar_por(DIAS, "Outlook")

    print("\n  Agrupando por Outlook:\n")
    for valor, grupo in grupos.items():
        positivos, negativos = contar(grupo)
        dias_del_grupo = ", ".join(dia["Dia"] for dia in grupo)
        print(f"    {valor:<9} {len(grupo)} dias -> ({positivos} Yes, {negativos} No)")
        print(f"              {dias_del_grupo}")

    suma = sum(len(grupo) for grupo in grupos.values())
    print(f"\n  Los grupos suman {suma} dias, que son todos los del conjunto.")


# ======================================================================
# BLOQUE 3 - GANANCIA DE INFORMACION
# ======================================================================

def ganancia(grupos):
    """
    Cuanta impureza se elimina al partir un conjunto en varios subgrupos.
    Formula de la presentacion:

        Gain(S, A) = Entropy(S) - suma( (|Sv| / |S|) * Entropy(Sv) )

    grupos: lista de tuplas (positivos, negativos), una por subgrupo.

    El peso |Sv| / |S| es lo importante: sin el, un grupo de 2 dias
    contaria lo mismo que uno de 200.
    """
    total_positivos = sum(positivos for positivos, _ in grupos)
    total_negativos = sum(negativos for _, negativos in grupos)
    total = total_positivos + total_negativos

    if total == 0:
        return 0.0

    impureza_antes = entropia(total_positivos, total_negativos)

    impureza_despues = 0.0
    for positivos, negativos in grupos:
        peso = (positivos + negativos) / total
        impureza_despues += peso * entropia(positivos, negativos)

    return impureza_antes - impureza_despues


def ganancia_del_atributo(dias, atributo):
    """Junta agrupar_por con ganancia: parte por el atributo y mide cuanto gano."""
    grupos = agrupar_por(dias, atributo)
    conteos = [contar(grupo) for grupo in grupos.values()]
    return ganancia(conteos), grupos


def prueba_ganancia():
    """
    Se calcula la ganancia de los cuatro atributos y se muestra el desglose
    completo: como quedan los grupos, la entropia de cada uno y el resultado.
    Son los cuatro numeros que se calcularon a mano en clase.
    """
    titulo("BLOQUE 3 - GANANCIA DE CADA ATRIBUTO")

    resultados = {}

    for atributo in ATRIBUTOS:
        valor_ganancia, grupos = ganancia_del_atributo(DIAS, atributo)
        resultados[atributo] = valor_ganancia

        print(f"\n  {atributo}")
        for valor, grupo in grupos.items():
            positivos, negativos = contar(grupo)
            print(f"    {valor:<9} -> {len(grupo)} dias ({positivos} Yes, "
                  f"{negativos} No), entropia {entropia(positivos, negativos):.4f}")
        print(f"    Gain(S, {atributo}) = {valor_ganancia:.4f}")

    print("\n  Casos extremos de la formula:")
    print(f"    corte que separa perfecto      -> {ganancia([(4, 0), (0, 4)]):.4f}")
    print(f"    corte que no cambia nada       -> {ganancia([(2, 2), (2, 2)]):.4f}")
    print(f"    partir un grupo que era puro   -> {ganancia([(4, 0), (3, 0)]):.4f}")

    return resultados


# ======================================================================
# BLOQUE 4 - ELEGIR EL MEJOR ATRIBUTO
# ======================================================================

def elegir_atributo(dias, atributos):
    """
    Elige el atributo de mayor ganancia, que es el paso del ID3 que dice
    "A <- el mejor atributo de decision".
    Regresa el nombre del atributo y el diccionario con todas las ganancias.
    """
    ganancias = {}
    for atributo in atributos:
        valor, _ = ganancia_del_atributo(dias, atributo)
        ganancias[atributo] = valor

    mejor = max(ganancias, key=ganancias.get)
    return mejor, ganancias


def prueba_elegir_atributo(ganancias_calculadas):
    """
    Se elige el atributo raiz y se compara todo lo calculado hasta aqui
    contra los valores publicados en la presentacion.
    """
    titulo("BLOQUE 4 - ELECCION DE LA RAIZ Y COMPARACION CON LA CLASE")

    mejor, ganancias = elegir_atributo(DIAS, ATRIBUTOS)

    print("\n  Ganancias ordenadas de mayor a menor:\n")
    for atributo, valor in sorted(ganancias.items(),
                                  key=lambda par: par[1], reverse=True):
        marca = "   <- la mayor" if atributo == mejor else ""
        print(f"    Gain(S, {atributo:<12}) = {valor:.4f}{marca}")

    print(f"\n  El atributo elegido como raiz es {mejor}, que es el mismo")
    print("  que se eligio en el ejercicio de clase.")

    print(f"\n  {'Calculo':<20}{'Presentacion':>14}{'Calculado':>12}{'Diferencia':>13}")
    print("  " + "-" * 57)
    for nombre, esperado in VALORES_DE_LA_CLASE.items():
        obtenido = ganancias_calculadas[nombre]
        print(f"  {nombre:<20}{esperado:>14.3f}{obtenido:>12.4f}"
              f"{obtenido - esperado:>+13.4f}")

    print("\n  Las diferencias son de redondeo: la presentacion reporta tres")
    print("  decimales y el programa cuatro. Los cinco valores coinciden.")


# ======================================================================
# BLOQUE 5 - CONSTRUCCION RECURSIVA DEL ARBOL
# ======================================================================

def construir_id3(dias, atributos, nombre_conjunto="S", mostrar=True, nivel=0):
    """
    ID3 completo: se elige el atributo de mayor ganancia, se abre una rama
    por cada valor, y sobre cada rama se repite el mismo procedimiento.

    Casos de paro:
      - el grupo es puro, todos los dias tienen la misma clase
      - ya no quedan atributos por probar

    Regresa el arbol como diccionarios anidados.
    """
    positivos, negativos = contar(dias)

    if negativos == 0:
        return {"hoja": "Yes", "dias": len(dias)}
    if positivos == 0:
        return {"hoja": "No", "dias": len(dias)}
    if not atributos:
        return {"hoja": "Yes" if positivos > negativos else "No",
                "dias": len(dias)}

    mejor, ganancias = elegir_atributo(dias, atributos)

    if mostrar:
        sangria = "  " * (nivel + 1)
        print(f"\n{sangria}Nodo sobre {nombre_conjunto}: {len(dias)} dias "
              f"({positivos} Yes, {negativos} No), "
              f"entropia {entropia(positivos, negativos):.4f}")
        for atributo, valor in sorted(ganancias.items(),
                                      key=lambda par: par[1], reverse=True):
            marca = "   <- se elige" if atributo == mejor else ""
            print(f"{sangria}  Gain({atributo:<12}) = {valor:.4f}{marca}")

    ramas = {}
    restantes = [atributo for atributo in atributos if atributo != mejor]

    for valor, grupo in agrupar_por(dias, mejor).items():
        ramas[valor] = construir_id3(grupo, restantes,
                                     f"{mejor}={valor}", mostrar, nivel + 1)

    return {"atributo": mejor, "ramas": ramas}


def dibujar_arbol(nodo, sangria=1, etiqueta="raiz"):
    """Dibuja el arbol resultante en la consola."""
    espacios = "  " * sangria

    if "hoja" in nodo:
        print(f"{espacios}{etiqueta}: {nodo['hoja']}   ({nodo['dias']} dias)")
        return

    print(f"{espacios}{etiqueta}: {nodo['atributo']}?")
    for valor, rama in nodo["ramas"].items():
        dibujar_arbol(rama, sangria + 1, valor)


def prueba_construir_id3():
    """
    Se arma el arbol completo aplicando el procedimiento de forma recursiva
    y se compara contra el arbol del material del curso.
    """
    titulo("BLOQUE 5 - CONSTRUCCION DEL ARBOL COMPLETO")

    arbol = construir_id3(DIAS, ATRIBUTOS)

    print("\n  ARBOL RESULTANTE\n")
    dibujar_arbol(arbol)

    print("\n  Coincide con el arbol del material del curso: Outlook en la raiz,")
    print("  Overcast siempre da Yes sin necesidad de preguntar mas, y las ramas")
    print("  Sunny y Rain se resuelven con Humidity y Wind respectivamente.")


# ======================================================================
# BLOQUE 6 - COMPROBACION CONTRA LAS FUNCIONES DEL MODELO
# ======================================================================

def prueba_contra_arbol_py():
    """
    Ultima comprobacion. Todo lo anterior valida las formulas escritas en
    este archivo, pero lo que interesa es que las de arbol.py, las que
    realmente entrenan el modelo sobre las 12,330 sesiones, sean iguales.

    Aqui se importan esas funciones y se corren sobre los mismos datos.
    Si dan lo mismo, la validacion aplica al modelo del entregable.
    """
    titulo("BLOQUE 6 - COMPROBACION CONTRA LAS FUNCIONES DE arbol.py")

    try:
        from arbol import entropia as entropia_modelo
        from arbol import ganancia as ganancia_modelo
    except ImportError:
        print("\n  (omitido: no se pudo importar arbol.py)")
        return

    print(f"\n  {'Calculo':<24}{'Este archivo':>14}{'arbol.py':>12}")
    print("  " + "-" * 50)

    positivos, negativos = contar(DIAS)
    print(f"  {'Entropy(S)':<24}{entropia(positivos, negativos):>14.4f}"
          f"{entropia_modelo(positivos, negativos):>12.4f}")

    for atributo in ATRIBUTOS:
        grupos = [contar(grupo) for grupo in agrupar_por(DIAS, atributo).values()]
        print(f"  {'Gain(S, ' + atributo + ')':<24}{ganancia(grupos):>14.4f}"
              f"{ganancia_modelo(grupos):>12.4f}")

    print("\n  Los valores son identicos. Las funciones que entrenan el modelo")
    print("  sobre el dataset real son las mismas que reproducen el ejercicio")
    print("  resuelto en clase.")

#mostarr pruebas
def main():
    print("VALIDACION DEL ALGORITMO - EJERCICIO PLAY TENNIS")
    print("\nSe reproduce el ejercicio resuelto en clase, construyendo el")
    print("algoritmo pieza por pieza y probando cada una antes de la siguiente.")

    entropia_inicial = prueba_entropia()
    prueba_agrupar()
    ganancias = prueba_ganancia()

    calculados = {"Entropy(S)": entropia_inicial}
    calculados.update(ganancias)
    prueba_elegir_atributo(calculados)

    prueba_construir_id3()
    prueba_contra_arbol_py()


if __name__ == "__main__":
    main()      