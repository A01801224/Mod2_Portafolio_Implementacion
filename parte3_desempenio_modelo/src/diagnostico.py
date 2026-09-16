"""
diagnostico.py
Diagnostico de bias, varianza y nivel de ajuste del modelo, comparando
su desempeño en entrenamiento contra su desempeño en validacion.

La logica es la clasica de bias/varianza:
- El desempeño en ENTRENAMIENTO indica que tanto bias tiene el modelo:
  si ni siquiera puede ajustar bien los datos que ya vio, el problema
  es que el modelo es demasiado simple para el patron (bias alto).
- La BRECHA entre entrenamiento y validacion indica la varianza: si el
  modelo memoriza detalles especificos del set de entrenamiento que no
  generalizan, la brecha es grande (varianza alta).
- El nivel de ajuste (underfit / fit / overfit) se deriva de combinar
  ambos diagnosticos.

Los umbrales usados abajo son un criterio razonable para este problema
(F1 con clases desbalanceadas, en un rango tipico de 0.0 a 0.7 para este
dataset), no una regla universal. Se documentan explicitamente para que
el diagnostico sea reproducible y no quede "a ojo".
"""

# Umbrales para diagnosticar BIAS a partir del F1 en entrenamiento.
# Si el modelo no logra un F1 razonable ni en los datos que ya vio,
# es señal de que le falta capacidad para capturar el patron (bias alto).
UMBRAL_BIAS_BAJO = 0.75    # F1 entrenamiento >= esto -> bias bajo
UMBRAL_BIAS_ALTO = 0.45    # F1 entrenamiento < esto -> bias alto
# Entre ambos umbrales -> bias medio

# Umbrales para diagnosticar VARIANZA a partir de la brecha F1 train - F1 val.
UMBRAL_VARIANZA_BAJA = 0.05   # brecha <= esto -> varianza baja
UMBRAL_VARIANZA_ALTA = 0.15   # brecha > esto -> varianza alta
# Entre ambos umbrales -> varianza media


def diagnosticar_bias(f1_entrenamiento):
    """Bajo / medio / alto, segun que tan bien ajusta el modelo sus propios
    datos de entrenamiento."""
    if f1_entrenamiento >= UMBRAL_BIAS_BAJO:
        return "bajo"
    elif f1_entrenamiento < UMBRAL_BIAS_ALTO:
        return "alto"
    else:
        return "medio"


def diagnosticar_varianza(f1_entrenamiento, f1_validacion):
    """Bajo / medio / alto, segun el tamano de la brecha entre entrenamiento
    y validacion."""
    brecha = f1_entrenamiento - f1_validacion
    if brecha <= UMBRAL_VARIANZA_BAJA:
        return "baja"
    elif brecha > UMBRAL_VARIANZA_ALTA:
        return "alta"
    else:
        return "media"


def diagnosticar_ajuste(bias, varianza):
    """
    Underfit / fit / overfit, combinando los dos diagnosticos anteriores:
    - Underfit: el modelo ni siquiera aprende bien el set de entrenamiento
      (bias alto), sin importar la varianza.
    - Overfit: el modelo aprende muy bien el entrenamiento (bias bajo o
      medio) pero no generaliza (varianza alta).
    - Fit: aprende bien Y generaliza bien (bias bajo/medio, varianza baja).
    - Los casos intermedios (bias medio + varianza media, por ejemplo) se
      reportan como "fit" con una nota, ya que no son un problema severo
      en ninguna de las dos direcciones.
    """
    if bias == "alto":
        return "underfit"
    if varianza == "alta":
        return "overfit"
    return "fit"


def diagnosticar(metricas_entrenamiento, metricas_validacion):
    """
    Corre el diagnostico completo a partir de los diccionarios de metricas
    (los que regresa calcular_metricas en metricas.py) de entrenamiento y
    validacion. Regresa un diccionario con los tres diagnosticos, los
    valores de F1 usados y la brecha, para poder imprimirlos o graficarlos.
    """
    f1_entrenamiento = metricas_entrenamiento["f1"]
    f1_validacion = metricas_validacion["f1"]
    brecha = f1_entrenamiento - f1_validacion

    bias = diagnosticar_bias(f1_entrenamiento)
    varianza = diagnosticar_varianza(f1_entrenamiento, f1_validacion)
    ajuste = diagnosticar_ajuste(bias, varianza)

    return {
        "f1_entrenamiento": f1_entrenamiento,
        "f1_validacion": f1_validacion,
        "brecha": brecha,
        "bias": bias,
        "varianza": varianza,
        "ajuste": ajuste
    }


def imprimir_diagnostico(diagnostico, titulo="Diagnostico"):
    """Imprime el diagnostico en un formato legible para la consola."""
    print(titulo)
    print("-" * len(titulo))
    print(f"  F1 entrenamiento: {diagnostico['f1_entrenamiento']:.4f}")
    print(f"  F1 validacion:    {diagnostico['f1_validacion']:.4f}")
    print(f"  Brecha:           {diagnostico['brecha']:+.4f}")
    print()
    print(f"  Bias:     {diagnostico['bias']}")
    print(f"  Varianza: {diagnostico['varianza']}")
    print(f"  Ajuste:   {diagnostico['ajuste']}")
    print()