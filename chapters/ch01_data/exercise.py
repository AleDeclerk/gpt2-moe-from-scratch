"""Capítulo 1: el pipeline de datos.

Escribí la clase y las dos funciones. Leé primero README.md.
"""

from __future__ import annotations

import torch


class CharTokenizer:
    """Un tokenizador que le da un entero a cada carácter distinto.

    Atributos:
        chars: la lista ordenada de los caracteres del vocabulario.
        stoi: un dict que lleva de un carácter a su entero.
        itos: un dict que lleva de un entero a su carácter.
    """

    def __init__(self, chars: list[str]) -> None:
        """Armá los dos mapas a partir de una lista ordenada de caracteres."""
        raise NotImplementedError("CharTokenizer.__init__")

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Armá un tokenizador con los caracteres que aparecen en `text`.

        El orden tiene que ser estable entre corridas: ordená los caracteres.
        """
        raise NotImplementedError("CharTokenizer.from_text")

    @property
    def vocab_size(self) -> int:
        """Devolvé la cantidad de tokens distintos."""
        raise NotImplementedError("CharTokenizer.vocab_size")

    def encode(self, text: str) -> list[int]:
        """Convertí un string en una lista de enteros."""
        raise NotImplementedError("CharTokenizer.encode")

    def decode(self, ids: list[int]) -> str:
        """Convertí una lista de enteros de vuelta a un string."""
        raise NotImplementedError("CharTokenizer.decode")


def train_val_split(
    data: torch.Tensor,
    val_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cortá la secuencia en una parte de entrenamiento y una de validación.

    Argumentos:
        data: un tensor 1-D de identificadores de tokens.
        val_fraction: la porción de los datos que va a validación, entre 0 y 1.

    Devuelve:
        Una tupla (train, val). Las dos partes son contiguas, y `val` es el
        final de la secuencia. Entre las dos tienen todos los elementos de
        `data`.

    Reglas:
        No mezcles nada. El README.md explica por qué un split aleatorio da un
        loss de validación demasiado bueno.
    """
    raise NotImplementedError("train_val_split")


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tomá un batch de bloques al azar, y los mismos corridos una posición.

    Argumentos:
        data: un tensor 1-D de identificadores de tokens.
        batch_size: la cantidad de bloques, B.
        block_size: la cantidad de tokens de cada bloque, T.
        generator: un torch.Generator para los offsets al azar, o None.

    Devuelve:
        Una tupla (x, y). Los dos tensores tienen shape (B, T) y dtype int64.
        y[b, t] es el token que viene justo después de x[b, t].

    Reglas:
        Usá torch.randint con el argumento `generator`, así una seed fija te da
        siempre el mismo batch.
        Ojo con el offset válido más grande: y lee una posición más allá de x.
    """
    raise NotImplementedError("get_batch")
