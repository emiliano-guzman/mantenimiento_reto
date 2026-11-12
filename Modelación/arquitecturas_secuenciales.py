"""Tres arquitecturas de dos salidas y una pérdida combinada, sin entrenamiento automático."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvolucionCausal(nn.Module):
    """Aplicar una convolución 1D que solo consulta la hora actual y las anteriores."""

    def __init__(self, entradas, salidas, kernel=3, dilatacion=1):
        super().__init__()
        self.relleno_izquierdo = (kernel - 1) * dilatacion
        self.conv = nn.Conv1d(entradas, salidas, kernel_size=kernel, dilation=dilatacion)

    def forward(self, x):

        # Rellenar solo por la izquierda conserva la longitud y evita consultar horas futuras.
        x = F.pad(x, (self.relleno_izquierdo, 0))
        return self.conv(x)


class BloqueTCN(nn.Module):
    """Convolución dilatada con una conexión residual para facilitar el aprendizaje."""

    def __init__(self, entradas, salidas, dilatacion, dropout):
        super().__init__()
        self.conv = ConvolucionCausal(entradas, salidas, kernel=3, dilatacion=dilatacion)
        self.activacion = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Ajustar canales solo si la entrada y la salida tienen tamaños distintos.
        self.residual = nn.Identity() if entradas == salidas else nn.Conv1d(entradas, salidas, kernel_size=1)

    def forward(self, x):
        salida = self.dropout(self.activacion(self.conv(x)))
        return self.activacion(salida + self.residual(x))


class ModeloDosSalidas(nn.Module):
    """Compartir embedding, unión de entradas estáticas y cabezas entre arquitecturas."""

    def __init__(self, numero_modelos, ocultas=32, embedding=4, dropout=0.1):
        super().__init__()

        # El código 0 queda reservado para modelos de máquina desconocidos.
        self.embedding_modelo = nn.Embedding(numero_modelos, embedding, padding_idx=0)
        self.combinar = nn.Sequential(
            nn.Linear(ocultas + embedding + 1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.salida_binaria = nn.Linear(32, 1)
        self.salida_multiclase = nn.Linear(32, 5)

    def forward(self, dinamicas, modelo, edad):

        # Cada codificador devuelve un estado por hora: B × L × ocultas.
        secuencia = self.codificar_secuencia(dinamicas)
        estado_final = secuencia[:, -1, :]
        representacion_modelo = self.embedding_modelo(modelo)

        # Unir la historia con el modelo de máquina y su antigüedad normalizada.
        juntas = torch.cat([estado_final, representacion_modelo, edad], dim=1)
        compartida = self.combinar(juntas)

        # Devolver logits, sin sigmoid ni softmax, para las funciones de pérdida estables.
        return {
            "binaria": self.salida_binaria(compartida),
            "multiclase": self.salida_multiclase(compartida),
        }


class ModeloLSTM(ModeloDosSalidas):
    """Una capa LSTM unidireccional; las entradas estáticas se unen al último estado."""

    def __init__(self, numero_modelos, canales=13, ocultas=32, embedding=4, dropout=0.1):
        super().__init__(numero_modelos, ocultas, embedding, dropout)
        self.lstm = nn.LSTM(canales, ocultas, num_layers=1, batch_first=True)

    def codificar_secuencia(self, dinamicas):
        estados, _ = self.lstm(dinamicas)
        return estados


class ModeloTCN(ModeloDosSalidas):
    """Siete bloques causales dilatados, sin reducir la longitud de la ventana."""

    def __init__(self, numero_modelos, canales=13, ocultas=32, embedding=4, dropout=0.1):
        super().__init__(numero_modelos, ocultas, embedding, dropout)
        self.dilataciones = [1, 2, 4, 8, 16, 32, 64]
        bloques = []
        entrada = canales
        for dilatacion in self.dilataciones:
            bloques.append(BloqueTCN(entrada, ocultas, dilatacion, dropout))
            entrada = ocultas
        self.bloques = nn.Sequential(*bloques)

        # Una convolución de kernel 3 por bloque: 1 + 2 × suma(dilataciones) = 255 horas.
        self.campo_receptivo = 1 + 2 * sum(self.dilataciones)

    def codificar_secuencia(self, dinamicas):

        # Conv1d espera B × canales × L; devolver luego el orden común B × L × ocultas.
        x = dinamicas.transpose(1, 2)
        x = self.bloques(x)
        return x.transpose(1, 2)


class ModeloCNNLSTM(ModeloDosSalidas):
    """Dos convoluciones causales extraen patrones locales; una LSTM recorre sus salidas."""

    def __init__(self, numero_modelos, canales=13, ocultas=32, embedding=4, dropout=0.1):
        super().__init__(numero_modelos, ocultas, embedding, dropout)
        self.cnn = nn.Sequential(
            ConvolucionCausal(canales, ocultas, kernel=3, dilatacion=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            ConvolucionCausal(ocultas, ocultas, kernel=3, dilatacion=2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(ocultas, ocultas, num_layers=1, batch_first=True)

    def codificar_secuencia(self, dinamicas):
        x = self.cnn(dinamicas.transpose(1, 2)).transpose(1, 2)
        estados, _ = self.lstm(x)
        return estados


class PerdidaCombinada(nn.Module):
    """Dar prioridad a la tarea binaria sin omitir el aprendizaje del componente."""

    def __init__(self, peso_binaria=0.6, peso_positivo=None, pesos_clase=None):
        super().__init__()
        if not 0 < peso_binaria < 1:
            raise ValueError("El peso binario debe estar entre 0 y 1, sin incluir los extremos.")
        self.peso_binaria = peso_binaria

        # Los pesos opcionales por clase se calcularán solo con entrenamiento en el próximo paso.
        self.binaria = nn.BCEWithLogitsLoss(pos_weight=peso_positivo)
        self.multiclase = nn.CrossEntropyLoss(weight=pesos_clase)

    def forward(self, salidas, lote):
        perdida_binaria = self.binaria(salidas["binaria"], lote["binaria"])
        perdida_multiclase = self.multiclase(salidas["multiclase"], lote["multiclase"])
        total = (self.peso_binaria * perdida_binaria
                 + (1 - self.peso_binaria) * perdida_multiclase)
        return {"total": total, "binaria": perdida_binaria, "multiclase": perdida_multiclase}


def crear_arquitecturas(numero_modelos):
    """Instanciar las tres alternativas con entradas y cabezas comunes."""
    return {
        "LSTM": ModeloLSTM(numero_modelos),
        "TCN": ModeloTCN(numero_modelos),
        "CNN-LSTM": ModeloCNNLSTM(numero_modelos),
    }
