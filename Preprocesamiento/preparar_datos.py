"""Preparación reproducible. Desde la raíz: python Preprocesamiento/preparar_datos.py."""
from pathlib import Path
import numpy as np
import pandas as pd

SENSORES = ["volt", "rotate", "pressure", "vibration"]
COMPONENTES = ["comp1", "comp2", "comp3", "comp4"]
ERRORES = ["error1", "error2", "error3", "error4", "error5"]
HORIZONTE = pd.Timedelta(hours=24)
PERIODOS = {
    "entrenamiento": (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-09-01")),
    "validación": (pd.Timestamp("2015-09-01"), pd.Timestamp("2015-11-01")),
    "prueba": (pd.Timestamp("2015-11-01"), pd.Timestamp("2016-01-01")),
}


def cargar_datos(ruta_repo):
    """Leer los cinco CSV y conservar solo los periodos necesarios."""
    archivos = {"telemetria": "PdM_telemetry.csv", "errores": "PdM_errors.csv",
                "mantenimiento": "PdM_maint.csv", "fallas": "PdM_failures.csv",
                "maquinas": "PdM_machines.csv"}
    datos = {}

    for nombre, archivo in archivos.items():
        fechas = [] if nombre == "maquinas" else ["datetime"]
        tabla = pd.read_csv(Path(ruta_repo) / "Datos" / archivo, parse_dates=fechas)

        # Mantener 2014 únicamente como historia de reemplazos; no incorporar 2016.
        if nombre != "maquinas":
            seleccion = tabla["datetime"] < PERIODOS["prueba"][1]
            if nombre != "mantenimiento":
                seleccion = seleccion & (tabla["datetime"] >= PERIODOS["entrenamiento"][0])
            tabla = tabla.loc[seleccion].copy()
            assert not tabla["datetime"].isna().any()
        datos[nombre] = tabla

    # Evitar cruces ambiguos y comprobar las claves de las fuentes.
    assert not datos["maquinas"]["machineID"].duplicated().any()
    assert not datos["telemetria"].duplicated(["machineID", "datetime"]).any()
    assert not datos["fallas"].duplicated(["machineID", "datetime", "failure"]).any()
    assert not datos["mantenimiento"].duplicated(["machineID", "datetime", "comp"]).any()
    assert set(datos["fallas"]["failure"]).issubset(COMPONENTES)
    assert set(datos["errores"]["errorID"]).issubset(ERRORES)
    ids = set(datos["maquinas"]["machineID"])
    for nombre in ["telemetria", "errores", "mantenimiento", "fallas"]:
        assert set(datos[nombre]["machineID"]).issubset(ids)
    return datos


def crear_variables(datos):
    """Integrar sensores, errores, reemplazos y catálogo con información histórica."""
    partes = []
    telemetria = datos["telemetria"].sort_values(["machineID", "datetime"])

    # Calcular antes de separar periodos: septiembre puede consultar historia de agosto.
    # Cada operación es histórica; calcular todas las fechas no permite mirar hacia adelante.
    for maquina_id, grupo in telemetria.groupby("machineID"):
        tabla = grupo.copy().reset_index(drop=True)
        horas = pd.DatetimeIndex(tabla["datetime"])
        diferencias = tabla["datetime"].diff().dropna()
        assert (diferencias == pd.Timedelta(hours=1)).all(), "Revisar huecos de telemetría."

        # Con continuidad horaria, 3 o 24 filas equivalen a las últimas 3 o 24 horas.
        # Incluir t supone que sus mediciones ya se observaron al emitir la predicción.
        for ventana in [3, 24]:
            for sensor in SENSORES:
                historial = tabla[sensor].rolling(ventana, min_periods=ventana)
                tabla[f"{sensor}_media_{ventana}h"] = historial.mean()
                tabla[f"{sensor}_desv_{ventana}h"] = historial.std(ddof=1)

        # Contar eventos por hora y tipo antes de unirlos; así no duplicamos la telemetría.
        eventos = datos["errores"].loc[datos["errores"]["machineID"] == maquina_id]
        conteos = eventos.groupby(["datetime", "errorID"]).size().unstack(fill_value=0)
        calendario = pd.date_range(PERIODOS["entrenamiento"][0], horas.max(), freq="h")
        conteos = conteos.reindex(index=calendario, columns=ERRORES, fill_value=0).fillna(0)
        acumulados = conteos.rolling(24, min_periods=1).sum()

        # La ventana de errores es (t − 24h, t]. La ausencia de un evento se cuenta como cero.
        # Esto presupone que el registro de errores está completo durante el periodo observado.
        for error in ERRORES:
            tabla[error + "_conteo_24h"] = acumulados.loc[horas, error].to_numpy().astype("int32")

        for componente in COMPONENTES:

            # Usar reemplazos de la misma máquina y componente, incluidos los de 2014.
            seleccion = ((datos["mantenimiento"]["machineID"] == maquina_id)
                         & (datos["mantenimiento"]["comp"] == componente))
            fechas = pd.DatetimeIndex(datos["mantenimiento"].loc[seleccion, "datetime"].sort_values())

            # side='left' excluye reemplazos de la propia hora: conservar solo fechas < t.
            posiciones = fechas.searchsorted(horas, side="left") - 1
            conocidos = posiciones >= 0
            ultimo = pd.Series(pd.NaT, index=tabla.index, dtype="datetime64[ns]")
            ultimo.loc[conocidos] = fechas.take(posiciones[conocidos]).to_numpy()
            tabla[componente + "_horas_desde_reemplazo"] = (tabla["datetime"] - ultimo).dt.total_seconds() / 3600

        partes.append(tabla)

    variables = pd.concat(partes, ignore_index=True)

    # El catálogo tiene una fila por máquina; añadir modelo y antigüedad sin multiplicar filas.
    variables = variables.merge(datos["maquinas"], on="machineID", how="left", validate="many_to_one")
    assert variables[["model", "age"]].notna().all().all()
    assert not variables.duplicated(["machineID", "datetime"]).any()
    return variables


def crear_etiquetas(horas, eventos):
    """Agregar las dos etiquetas y devolver indicadores temporales para el resumen."""
    partes = []
    auditoria = []
    for maquina_id, grupo in horas.groupby("machineID"):
        tabla = grupo.copy().reset_index(drop=True)
        consultas = pd.DatetimeIndex(tabla["datetime"])
        siguientes = pd.DataFrame(index=tabla.index)

        for componente in COMPONENTES:

            # Buscar la primera falla estrictamente posterior a cada hora t.
            seleccion = (eventos["machineID"] == maquina_id) & (eventos["failure"] == componente)
            fechas = pd.DatetimeIndex(eventos.loc[seleccion, "datetime"].sort_values())
            posiciones = fechas.searchsorted(consultas, side="right")
            existe = posiciones < len(fechas)
            futura = pd.Series(pd.NaT, index=tabla.index, dtype="datetime64[ns]")
            futura.loc[existe] = fechas.take(posiciones[existe]).to_numpy()

            # Conservar solo fallas en (t, t + 24h], incluido el extremo de 24 horas.
            dentro = (futura > tabla["datetime"]) & (futura <= tabla["datetime"] + HORIZONTE)
            siguientes[componente] = futura.where(dentro)

        primera_fecha = siguientes.min(axis=1)
        indicadores = siguientes.notna()
        primeros = siguientes.eq(primera_fecha, axis=0)
        tabla["etiqueta_binaria"] = primera_fecha.notna().astype("int8")
        tabla["etiqueta_multiclase"] = "ninguno"

        # Elegir la primera falla; en simultáneas, el orden comp1–comp4 resuelve el empate.
        for componente in COMPONENTES:
            elegir = primeros[componente] & (tabla["etiqueta_multiclase"] == "ninguno")
            tabla.loc[elegir, "etiqueta_multiclase"] = componente

        # Las columnas auxiliares se usan solo en resúmenes, no se exportan como entradas.
        indicadores["varios_componentes"] = siguientes.notna().sum(axis=1) > 1
        indicadores["empate_primera_falla"] = primeros.sum(axis=1) > 1
        partes.append(tabla)
        auditoria.append(indicadores)

    return pd.concat(partes, ignore_index=True), pd.concat(auditoria, ignore_index=True)


def preparar_conjuntos(variables, fallas):
    """Separar periodos, aplicar el margen y etiquetar con eventos de cada periodo."""
    conjuntos = {}
    resumen_binario = []
    resumen_clases = []
    resumen_componentes = []
    resumen_particion = []
    resumen_empates = []

    for nombre, (inicio, limite) in PERIODOS.items():
        seleccion = variables["datetime"].between(inicio, limite, inclusive="left")
        periodo = variables.loc[seleccion].copy()

        # Retirar las últimas 24 horas del periodo; no etiquetarlas como sanas por falta de futuro.
        completas = periodo["datetime"] + HORIZONTE < limite
        utiles = periodo.loc[completas].copy()
        eventos = fallas.loc[fallas["datetime"].between(inicio, limite, inclusive="left")]
        tabla, auditoria = crear_etiquetas(utiles, eventos)
        assert (tabla["datetime"] + HORIZONTE < limite).all()
        assert (tabla["etiqueta_binaria"] == (tabla["etiqueta_multiclase"] != "ninguno").astype(int)).all()
        conjuntos[nombre] = tabla
        resumen_particion.append({"conjunto": nombre, "filas": len(tabla), "excluidas_frontera": int((~completas).sum()),
                                  "primera_hora": tabla["datetime"].min(), "ultima_hora": tabla["datetime"].max()})

        # Prueba se guarda, pero no se exploran sus etiquetas ni se calculan métricas de modelos.
        if nombre == "prueba":
            continue

        total = len(tabla)
        positivos = int(tabla["etiqueta_binaria"].sum())
        resumen_binario.append({"conjunto": nombre, "filas": total, "positivos": positivos,
                                "negativos": total - positivos, "porcentaje_positivos": 100 * positivos / total})
        for clase in ["ninguno"] + COMPONENTES:
            cantidad = int((tabla["etiqueta_multiclase"] == clase).sum())
            resumen_clases.append({"conjunto": nombre, "clase": clase, "horas": cantidad, "porcentaje": 100 * cantidad / total})
        for componente in COMPONENTES:
            cantidad = int(auditoria[componente].sum())
            resumen_componentes.append({"conjunto": nombre, "componente": componente, "horas_positivas": cantidad,
                                       "porcentaje": 100 * cantidad / total})
        resumen_empates.append({"conjunto": nombre, "varios_componentes": int(auditoria["varios_componentes"].sum()),
                               "empates_primera_falla": int(auditoria["empate_primera_falla"].sum())})

    resumenes = {"particion": pd.DataFrame(resumen_particion), "binario": pd.DataFrame(resumen_binario),
                "clases": pd.DataFrame(resumen_clases), "componentes": pd.DataFrame(resumen_componentes),
                "empates": pd.DataFrame(resumen_empates)}
    return conjuntos, resumenes


def guardar_conjuntos(conjuntos, ruta_repo):
    """Guardar únicamente tres CSV finales, con claves, variables y las dos etiquetas."""
    destino = Path(ruta_repo) / "Datos" / "Datos_preparados"
    destino.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in conjuntos.items():
        tabla.to_csv(destino / (nombre + ".csv"), index=False)
        print("Guardado:", nombre + ".csv")


def ejecutar_pipeline(ruta_repo):
    """Reproducir toda la preparación desde los cinco archivos originales."""
    datos = cargar_datos(ruta_repo)
    variables = crear_variables(datos)
    conjuntos, resumenes = preparar_conjuntos(variables, datos["fallas"])
    guardar_conjuntos(conjuntos, ruta_repo)
    return conjuntos, resumenes


if __name__ == "__main__":
    # La ubicación del script permite ejecutarlo desde cualquier carpeta.
    ejecutar_pipeline(Path(__file__).resolve().parent.parent)
