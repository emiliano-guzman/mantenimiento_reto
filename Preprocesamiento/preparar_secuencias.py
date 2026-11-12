"""Preparación compartida de ventanas; no modifica CSV ni entrena modelos.

Uso desde otro notebook:
    preparacion = preparar_secuencias(ruta_repo)
    datos_entrenamiento = preparacion["datasets"][24]["entrenamiento"]

Desde la raíz del repositorio:
    python Preprocesamiento/preparar_secuencias.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

inicio = pd.Timestamp("2015-01-01")
limite_entrenamiento = pd.Timestamp("2015-09-01")
limite_validacion = pd.Timestamp("2015-11-01")
LONGITUDES = [24, 72, 168]
SENSORES = ["volt", "rotate", "pressure", "vibration"]
ERRORES = ["error1", "error2", "error3", "error4", "error5"]
COMPONENTES = ["comp1", "comp2", "comp3", "comp4"]
CLASES = ["ninguno", "comp1", "comp2", "comp3", "comp4"]
mapa_clases = {nombre: numero for numero, nombre in enumerate(CLASES)}
SEMILLA = 42

canales_errores = [error + "_horario" for error in ERRORES]
canales_reemplazos = [componente + "_horas_desde_reemplazo" for componente in COMPONENTES]
canales = SENSORES + canales_errores + canales_reemplazos
continuos = SENSORES + canales_reemplazos
claves = ["machineID", "datetime"]

def cargar_fuentes(ruta_repo):
    """Leer historia y objetivos de entrenamiento y validación; no abrir prueba."""
    ruta_datos = Path(ruta_repo) / "Datos"
    ruta_preparados = ruta_datos / "Datos_preparados"

    # Leer solo claves y etiquetas, sin cargar las variables de la línea base ni abrir prueba.
    columnas_objetivos = ["machineID", "datetime", "etiqueta_binaria", "etiqueta_multiclase"]
    objetivos = {}
    for nombre, archivo in [("entrenamiento", "entrenamiento.csv"), ("validación", "validación.csv")]:
        objetivos[nombre] = pd.read_csv(ruta_preparados / archivo, usecols=columnas_objetivos, parse_dates=["datetime"])

    # Conservar historia anterior a noviembre. El mantenimiento de 2014 es válido como antecedente.
    archivos = {"telemetria": "PdM_telemetry.csv", "errores": "PdM_errors.csv", "mantenimiento": "PdM_maint.csv"}
    datos = {}
    for nombre, archivo in archivos.items():
        tabla = pd.read_csv(ruta_datos / archivo, parse_dates=["datetime"])
        seleccion = tabla["datetime"] < limite_validacion
        if nombre != "mantenimiento":
            seleccion = seleccion & (tabla["datetime"] >= inicio)
        datos[nombre] = tabla.loc[seleccion].copy()
        del tabla
    maquinas = pd.read_csv(ruta_datos / "PdM_machines.csv")

    # Mantener las fronteras de las etiquetas y sus respuestas sin cambios.
    for nombre, tabla in objetivos.items():
        comienzo = inicio if nombre == "entrenamiento" else limite_entrenamiento
        limite = limite_entrenamiento if nombre == "entrenamiento" else limite_validacion
        assert tabla["datetime"].between(comienzo, limite, inclusive="left").all()
        assert (tabla["datetime"] + pd.Timedelta(hours=24) < limite).all()
        assert not tabla.duplicated(["machineID", "datetime"]).any()
        assert tabla[columnas_objetivos].notna().all().all()
        assert set(tabla["etiqueta_multiclase"]).issubset(CLASES)
        assert (tabla["etiqueta_binaria"] == (tabla["etiqueta_multiclase"] != "ninguno").astype(int)).all()
    return datos, maquinas, objetivos

def construir_historia(datos, maquinas):
    """Reconstruir 13 canales horarios sin mezclar máquinas ni consultar eventos futuros."""
    partes = []

    # Ordenar las mediciones permite revisar continuidad y consultar historia por máquina.
    telemetria = datos["telemetria"].sort_values(["machineID", "datetime"])
    assert not telemetria.duplicated(["machineID", "datetime"]).any()
    assert not maquinas["machineID"].duplicated().any()
    for nombre, tabla in datos.items():
        assert set(tabla["machineID"]).issubset(set(maquinas["machineID"]))
    assert set(datos["errores"]["errorID"]).issubset(ERRORES)

    for maquina_id, grupo in telemetria.groupby("machineID"):
        tabla = grupo.copy().reset_index(drop=True)
        horas = pd.DatetimeIndex(tabla["datetime"])
        assert (tabla["datetime"].diff().dropna() == pd.Timedelta(hours=1)).all()

        # Agregar primero los eventos para no duplicar filas de telemetría al unirlos.
        errores = datos["errores"].loc[datos["errores"]["machineID"] == maquina_id]
        conteos = errores.groupby(["datetime", "errorID"]).size().unstack(fill_value=0)
        conteos = conteos.reindex(index=horas, columns=ERRORES, fill_value=0).fillna(0)
        for error in ERRORES:
            tabla[error + "_horario"] = (conteos[error].to_numpy() > 0).astype("int8")

        for componente in COMPONENTES:

            # Buscar reemplazos solo de esta máquina y componente, incluyendo 2014.
            seleccion = ((datos["mantenimiento"]["machineID"] == maquina_id)
                         & (datos["mantenimiento"]["comp"] == componente))
            fechas = pd.DatetimeIndex(datos["mantenimiento"].loc[seleccion, "datetime"].sort_values())
            posiciones = fechas.searchsorted(horas, side="left") - 1
            conocidos = posiciones >= 0
            ultimo = pd.Series(pd.NaT, index=tabla.index, dtype="datetime64[ns]")
            ultimo.loc[conocidos] = fechas.take(posiciones[conocidos]).to_numpy()
            tabla[componente + "_horas_desde_reemplazo"] = (tabla["datetime"] - ultimo).dt.total_seconds() / 3600

        partes.append(tabla[["machineID", "datetime"] + canales])

    historia = pd.concat(partes, ignore_index=True)
    assert len(canales) == 13
    assert not historia.duplicated(["machineID", "datetime"]).any()
    assert historia[canales_errores].isin([0, 1]).all().all()
    assert not np.isinf(historia[continuos].to_numpy()).any()
    return historia

def normalizar_historia(historia, objetivos):
    """Aprender imputación y escalas solo con las claves de entrenamiento."""
    # Encontrar exactamente las filas de historia que corresponden a las claves de entrenamiento.
    claves = ["machineID", "datetime"]
    historia_indice = historia.set_index(claves).sort_index()
    indice_ajuste = pd.MultiIndex.from_frame(objetivos["entrenamiento"][claves])
    base_ajuste = historia_indice.reindex(indice_ajuste)
    assert base_ajuste[canales_errores].notna().all().all()
    assert base_ajuste[continuos].notna().any().all()

    # Aprender medianas y escalas exclusivamente con las horas de entrenamiento.
    imputador = SimpleImputer(strategy="median")
    escalador = StandardScaler()
    ajuste_imputado = imputador.fit_transform(base_ajuste[continuos])
    escalador.fit(ajuste_imputado)

    # Aplicar lo aprendido a la historia sin volver a ajustar en validación.
    historia_normalizada = historia.copy()
    valores_imputados = imputador.transform(historia[continuos])
    historia_normalizada[continuos] = escalador.transform(valores_imputados)
    assert np.isfinite(historia_normalizada[canales].to_numpy()).all()
    assert historia_normalizada[canales_errores].isin([0, 1]).all().all()
    assert np.allclose(imputador.statistics_, base_ajuste[continuos].median().to_numpy())

    estadisticas = pd.DataFrame({"canal": continuos, "faltantes_ajuste": base_ajuste[continuos].isna().sum().to_numpy(),
                                "mediana_entrenamiento": imputador.statistics_, "media_entrenamiento": escalador.mean_,
                                "desviacion_entrenamiento": escalador.scale_})
    return historia_normalizada, imputador, escalador, estadisticas

def preparar_estaticas(maquinas, objetivos):
    """Preparar el vocabulario del modelo y la antigüedad con máquinas de entrenamiento."""
    ids_entrenamiento = objetivos["entrenamiento"]["machineID"].unique()
    catalogo_ajuste = maquinas.loc[maquinas["machineID"].isin(ids_entrenamiento)].copy()
    assert catalogo_ajuste[["model", "age"]].notna().all().all()
    assert (catalogo_ajuste["age"] >= 0).all()

    # Mantener un mapa fijo; no agregar categorías mirando validación.
    modelos_conocidos = sorted(catalogo_ajuste["model"].unique())
    mapa_modelos = {nombre: numero + 1 for numero, nombre in enumerate(modelos_conocidos)}
    escalador_edad = StandardScaler().fit(catalogo_ajuste[["age"]])
    catalogo = maquinas.copy()
    catalogo["modelo_codigo"] = catalogo["model"].map(mapa_modelos).fillna(0).astype(int)
    catalogo["edad_normalizada"] = escalador_edad.transform(catalogo[["age"]]).ravel()
    catalogo = catalogo.set_index("machineID")
    return catalogo, mapa_modelos, escalador_edad

def crear_series(historia_normalizada, catalogo):
    """Guardar la historia una vez por máquina para recuperar ventanas sin duplicarla."""
    # Guardar por máquina las señales, las fechas y las entradas estáticas ya preparadas.
    series_maquina = {}
    for maquina_id, grupo in historia_normalizada.groupby("machineID"):
        grupo = grupo.sort_values("datetime").reset_index(drop=True)
        series_maquina[maquina_id] = {
            "horas": pd.DatetimeIndex(grupo["datetime"]),
            "valores": grupo[canales].to_numpy(dtype="float32"),
            "modelo": int(catalogo.loc[maquina_id, "modelo_codigo"]),
            "edad": float(catalogo.loc[maquina_id, "edad_normalizada"]),
        }
    return series_maquina

def localizar_anclas(tabla_objetivos, longitud, series_maquina):
    """Conservar anclas originales con historia suficiente, sin modificar sus etiquetas."""
    partes = []
    for maquina_id, grupo in tabla_objetivos.groupby("machineID"):
        horas = series_maquina[maquina_id]["horas"]

        # get_indexer ubica la hora exacta; −1 significaría que no existe en la historia.
        posiciones = horas.get_indexer(pd.DatetimeIndex(grupo["datetime"]))
        assert (posiciones >= 0).all()
        tabla = grupo.copy()
        tabla["posicion_final"] = posiciones

        # Como la continuidad horaria ya se comprobó, basta pedir L filas disponibles.
        tabla = tabla.loc[tabla["posicion_final"] >= longitud - 1].copy()
        tabla["clase_codigo"] = tabla["etiqueta_multiclase"].map(mapa_clases).astype(int)
        partes.append(tabla)
    return pd.concat(partes, ignore_index=True)


def crear_anclas(objetivos, series_maquina, longitudes=LONGITUDES):
    """Identificar ventanas completas y conservar las mismas horas de validación."""
    anclas = {}
    revision_ventanas = []
    for longitud in longitudes:
        anclas[longitud] = {}
        for nombre, tabla in objetivos.items():
            conservadas = localizar_anclas(tabla, longitud, series_maquina)
            anclas[longitud][nombre] = conservadas
            revision_ventanas.append({"conjunto": nombre, "L_horas": longitud, "anclas_originales": len(tabla),
                "ventanas_completas": len(conservadas), "excluidas_sin_historia": len(tabla) - len(conservadas),
                "primera_ancla": conservadas["datetime"].min(), "ultima_ancla": conservadas["datetime"].max()})


    referencia = anclas[max(longitudes)]["validación"][claves].sort_values(claves).reset_index(drop=True)
    for longitud in longitudes:
        comparar = anclas[longitud]["validación"][claves].sort_values(claves).reset_index(drop=True)
        pd.testing.assert_frame_equal(referencia, comparar)
    return anclas, pd.DataFrame(revision_ventanas)

class VentanasMaquina(Dataset):
    """Recuperar ventanas al solicitarlas, sin copiarlas todas en memoria."""

    def __init__(self, tabla_anclas, longitud, series):
        self.longitud = longitud
        self.series = series
        self.ids = tabla_anclas["machineID"].to_numpy()
        self.finales = tabla_anclas["posicion_final"].to_numpy()
        self.binarias = tabla_anclas["etiqueta_binaria"].to_numpy(dtype="float32")
        self.clases = tabla_anclas["clase_codigo"].to_numpy(dtype="int64")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, indice):
        maquina_id = self.ids[indice]
        serie = self.series[maquina_id]
        final = int(self.finales[indice])
        inicio_ventana = final - self.longitud + 1

        # La rebanada acaba en final + 1 porque Python excluye el extremo derecho.
        ventana = serie["valores"][inicio_ventana:final + 1]
        return {
            "dinamicas": torch.from_numpy(ventana),
            "modelo": torch.tensor(serie["modelo"], dtype=torch.long),
            "edad": torch.tensor([serie["edad"]], dtype=torch.float32),
            "binaria": torch.tensor([self.binarias[indice]], dtype=torch.float32),
            "multiclase": torch.tensor(self.clases[indice], dtype=torch.long),
        }




def crear_datasets(anclas, series_maquina):
    """Crear vistas de ventanas por longitud y conjunto sin materializarlas todas."""
    datasets = {}
    for longitud, conjuntos in anclas.items():
        datasets[longitud] = {}
        for nombre, tabla in conjuntos.items():
            datasets[longitud][nombre] = VentanasMaquina(tabla, longitud, series_maquina)
    return datasets


def preparar_secuencias(ruta_repo):
    """Ejecutar la preparación completa en memoria para los próximos modelos."""
    datos, maquinas, objetivos = cargar_fuentes(ruta_repo)
    historia = construir_historia(datos, maquinas)
    normalizada, imputador, escalador, estadisticas = normalizar_historia(historia, objetivos)
    catalogo, mapa_modelos, escalador_edad = preparar_estaticas(maquinas, objetivos)
    series = crear_series(normalizada, catalogo)
    anclas, resumen = crear_anclas(objetivos, series)
    datasets = crear_datasets(anclas, series)

    # Devolver datos para los modelos y los parámetros que deben reutilizarse sin reajustar.
    # No devolver copias de todas las tablas intermedias evita ocupar memoria innecesaria.
    return {
        "datasets": datasets, "anclas": anclas, "objetivos": objetivos,
        "canales": list(canales), "numero_codigos_modelo": len(mapa_modelos) + 1,
        "mapa_modelos": mapa_modelos, "imputador": imputador, "escalador": escalador,
        "escalador_edad": escalador_edad, "estadisticas": estadisticas,
        "resumen_ventanas": resumen,
    }


if __name__ == "__main__":
    # Esta ejecución comprueba la preparación, pero no crea un CSV ni deja variables
    # disponibles en otra sesión. Para entrenar, importar la función en el notebook del modelo.
    resultado = preparar_secuencias(Path(__file__).resolve().parent.parent)
    print(resultado["resumen_ventanas"].to_string(index=False))
    print("Preparación terminada en memoria. No se exportaron archivos ni se entrenaron modelos.")
