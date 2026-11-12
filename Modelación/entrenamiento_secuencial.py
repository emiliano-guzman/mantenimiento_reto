"""Comparar pérdidas con datos completos; las salidas permanecen en memoria."""
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, f1_score
from arquitecturas_secuenciales import ModeloLSTM, ModeloTCN, ModeloCNNLSTM


class PerdidaDesbalanceada(nn.Module):
    """Comparar ponderada y focal manteniendo los mismos pesos de clase en ambas."""

    def __init__(self, pesos_binarios, pesos_multiclase, estrategia, peso_binaria=0.60):
        super().__init__()
        if estrategia not in ["ponderada", "focal"]:
            raise ValueError("Estrategia desconocida.")
        self.gamma = 0.0 if estrategia == "ponderada" else 2.0
        self.peso_binaria = peso_binaria
        self.register_buffer("pesos_binarios", torch.as_tensor(pesos_binarios, dtype=torch.float32))
        self.register_buffer("pesos_multiclase", torch.as_tensor(pesos_multiclase, dtype=torch.float32))

    def forward(self, salidas, lote):
        objetivo_bin = lote["binaria"]
        objetivo_multi = lote["multiclase"]

        # Calcular BCE estable por ejemplo y seleccionar el peso de su clase real.
        bce = F.binary_cross_entropy_with_logits(salidas["binaria"], objetivo_bin, reduction="none")
        pesos_b = self.pesos_binarios[objetivo_bin.long()]
        pt_bin = torch.exp(-bce)
        binaria = (pesos_b * (1 - pt_bin).pow(self.gamma) * bce).mean()

        # Obtener la probabilidad de la clase verdadera sin aplicar softmax dos veces.
        log_probabilidades = F.log_softmax(salidas["multiclase"], dim=1)
        log_pt = log_probabilidades.gather(1, objetivo_multi.unsqueeze(1)).squeeze(1)
        pt_multi = log_pt.exp()
        multi = (-self.pesos_multiclase[objetivo_multi] * (1 - pt_multi).pow(self.gamma) * log_pt).mean()
        total = self.peso_binaria * binaria + (1 - self.peso_binaria) * multi
        return {"total": total, "binaria": binaria, "multiclase": multi}


def fijar_semilla(semilla):
    """Repetir inicialización, dropout y orden de lotes en cada ensayo."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True)


def indices_comunes(preparacion):
    """Usar exactamente las mismas anclas, con historia para 168h, en todos los ensayos."""
    comunes = preparacion["anclas"][168]["entrenamiento"][["machineID", "datetime"]]
    indice_comun = pd.MultiIndex.from_frame(comunes)
    indices = {}
    for longitud in [24, 72, 168]:
        tabla = preparacion["anclas"][longitud]["entrenamiento"]
        indice = pd.MultiIndex.from_frame(tabla[["machineID", "datetime"]])
        posiciones = indice.get_indexer(indice_comun)
        assert (posiciones >= 0).all()
        indices[longitud] = posiciones
    return indices


def calcular_pesos(tabla):
    """Pesos inversos a frecuencia, normalizados para que su promedio de entrenamiento sea 1."""
    n = len(tabla)
    conteo_bin = np.bincount(tabla["etiqueta_binaria"].to_numpy(dtype=int), minlength=2)
    conteo_multi = np.bincount(tabla["clase_codigo"].to_numpy(dtype=int), minlength=5)
    assert (conteo_bin > 0).all() and (conteo_multi > 0).all()
    return n / (2 * conteo_bin), n / (5 * conteo_multi)


def predecir(modelo, cargador, criterio=None):
    """Evaluar sin gradientes y conservar el orden de todas las horas de validación."""
    modelo.eval()
    probabilidades = []
    multiclase = []
    reales_binarios = []
    reales_multiclase = []
    sumas = {"total": 0.0, "binaria": 0.0, "multiclase": 0.0}
    total = 0
    with torch.no_grad():
        for lote in cargador:
            salidas = modelo(lote["dinamicas"], lote["modelo"], lote["edad"])
            probabilidades.append(torch.sigmoid(salidas["binaria"]).squeeze(1).numpy())
            multiclase.append(torch.softmax(salidas["multiclase"], dim=1).numpy())
            reales_binarios.append(lote["binaria"].squeeze(1).numpy())
            reales_multiclase.append(lote["multiclase"].numpy())
            B = len(lote["modelo"])
            total += B
            if criterio is not None:
                perdidas = criterio(salidas, lote)
                for nombre in sumas:
                    sumas[nombre] += perdidas[nombre].item() * B
    resultado = {"prob_binaria": np.concatenate(probabilidades), "prob_multiclase": np.concatenate(multiclase),
                 "y_binaria": np.concatenate(reales_binarios), "y_multiclase": np.concatenate(reales_multiclase)}
    resultado["perdidas"] = {nombre: valor / total for nombre, valor in sumas.items()}
    return resultado


def error_calibracion(y, p, numero_bins=10):
    """ECE descriptivo: distancia entre probabilidad media y frecuencia positiva en 10 intervalos fijos."""
    grupos = np.minimum((p * numero_bins).astype(int), numero_bins - 1)
    ece = 0.0
    for grupo in range(numero_bins):
        seleccion = grupos == grupo
        if seleccion.any():
            ece += seleccion.mean() * abs(p[seleccion].mean() - y[seleccion].mean())
    return float(ece)


def resumir_predicciones(predicciones):
    """Métricas de discriminación y calibración, siempre sobre validación sin remuestreo."""
    y = predicciones["y_binaria"]
    p = predicciones["prob_binaria"]
    return {"PR_AUC_AP": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1]),
            "ECE_10_bins": error_calibracion(y, p),
            "F1_macro_5_clases": f1_score(predicciones["y_multiclase"], predicciones["prob_multiclase"].argmax(axis=1),
                                            labels=range(5), average="macro", zero_division=0)}


def entrenar_ensayo(preparacion, indices, arquitectura, longitud, estrategia, configuracion):
    """Ajustar con todas las anclas comunes y restaurar la mejor época por PR-AUC de validación."""
    fijar_semilla(configuracion["semilla"])
    constructores = {"LSTM": ModeloLSTM, "TCN": ModeloTCN, "CNN-LSTM": ModeloCNNLSTM}
    modelo = constructores[arquitectura](preparacion["numero_codigos_modelo"], dropout=configuracion["dropout"])
    tabla_pesos = preparacion["anclas"][168]["entrenamiento"]
    pesos_binarios, pesos_multi = calcular_pesos(tabla_pesos)
    criterio = PerdidaDesbalanceada(pesos_binarios, pesos_multi, estrategia)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=configuracion["learning_rate"],
                                    weight_decay=configuracion["weight_decay"])

    # El muestreo solo permuta entrenamiento; no elimina filas ni cambia su prevalencia.
    generador = torch.Generator().manual_seed(configuracion["semilla"])
    entrenamiento = Subset(preparacion["datasets"][longitud]["entrenamiento"], indices[longitud])
    validacion = preparacion["datasets"][longitud]["validación"]
    cargador_train = DataLoader(entrenamiento, batch_size=configuracion["batch_size"], shuffle=True,
                               generator=generador, num_workers=0)
    cargador_val = DataLoader(validacion, batch_size=configuracion["batch_size"], shuffle=False, num_workers=0)
    mejor_ap = -np.inf
    mejor_estado = None
    mejor_epoca = 0
    sin_mejora = 0
    historial = []
    inicio_ensayo = time.perf_counter()
    identificador = f"{arquitectura} | L={longitud} | {estrategia}"
    print("INICIO", identificador, flush=True)

    for epoca in range(1, configuracion["max_epocas"] + 1):
        inicio_epoca = time.perf_counter()
        modelo.train()
        sumas = {"total": 0.0, "binaria": 0.0, "multiclase": 0.0}
        procesados = 0
        for numero_lote, lote in enumerate(cargador_train, start=1):
            optimizador.zero_grad(set_to_none=True)
            salidas = modelo(lote["dinamicas"], lote["modelo"], lote["edad"])
            perdidas = criterio(salidas, lote)
            assert torch.isfinite(perdidas["total"])
            perdidas["total"].backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizador.step()
            B = len(lote["modelo"])
            procesados += B
            for nombre in sumas:
                sumas[nombre] += perdidas[nombre].item() * B
            if numero_lote % 300 == 0:
                print(identificador, "época", epoca, "lotes", numero_lote, "/", len(cargador_train), flush=True)
        assert procesados == len(entrenamiento)

        evaluacion = predecir(modelo, cargador_val, criterio)
        ap = average_precision_score(evaluacion["y_binaria"], evaluacion["prob_binaria"])
        fila = {"arquitectura": arquitectura, "L": longitud, "estrategia": estrategia, "epoca": epoca,
                "PR_AUC_validacion": ap, "segundos": time.perf_counter() - inicio_epoca}
        for nombre in sumas:
            fila["train_" + nombre] = sumas[nombre] / procesados
            fila["val_" + nombre] = evaluacion["perdidas"][nombre]
        historial.append(fila)
        print(identificador, "FIN época", epoca, "PR-AUC", round(ap, 6), "segundos", round(fila["segundos"], 1), flush=True)

        # Guardar siempre el mejor valor real; usar min_delta para decidir la paciencia.
        avance_suficiente = ap > mejor_ap + configuracion["min_delta"]
        if ap > mejor_ap:
            mejor_ap = ap
            mejor_estado = copy.deepcopy(modelo.state_dict())
            mejor_epoca = epoca
        sin_mejora = 0 if avance_suficiente else sin_mejora + 1
        if sin_mejora >= configuracion["paciencia"]:
            print("Early stopping:", identificador, flush=True)
            break

    modelo.load_state_dict(mejor_estado)
    predicciones = predecir(modelo, cargador_val)
    resumen = {"arquitectura": arquitectura, "L": longitud, "estrategia": estrategia,
               "epocas_ejecutadas": len(historial), "mejor_epoca": mejor_epoca,
               "filas_entrenamiento": len(entrenamiento), "filas_validacion": len(validacion),
               "segundos_totales": time.perf_counter() - inicio_ensayo}
    resumen.update(resumir_predicciones(predicciones))
    return {"modelo": modelo, "resumen": resumen, "historial": pd.DataFrame(historial), "predicciones": predicciones}


def ejecutar_comparacion(preparacion, configuracion):
    """Ocho ensayos: seis para arquitecturas/pérdidas a 24h y dos para la mejor pareja a 72/168h."""
    torch.set_num_threads(configuracion["hilos"])
    indices = indices_comunes(preparacion)
    resultados = {}
    for arquitectura in ["LSTM", "TCN", "CNN-LSTM"]:
        for estrategia in ["ponderada", "focal"]:
            clave = (arquitectura, 24, estrategia)
            resultados[clave] = entrenar_ensayo(preparacion, indices, *clave, configuracion)

    # Seleccionar solo con validación; en empate, conservar el primer ensayo del orden declarado.
    mejor_clave = max(resultados, key=lambda clave: resultados[clave]["resumen"]["PR_AUC_AP"])
    arquitectura, _, estrategia = mejor_clave
    print("SEGUNDA ETAPA: longitudes con", arquitectura, estrategia, flush=True)
    for longitud in [72, 168]:
        clave = (arquitectura, longitud, estrategia)
        resultados[clave] = entrenar_ensayo(preparacion, indices, *clave, configuracion)
    return resultados
