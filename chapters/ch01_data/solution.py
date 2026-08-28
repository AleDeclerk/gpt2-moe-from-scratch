"""Capítulo 1: código de referencia."""

from __future__ import annotations

import torch


class CharTokenizer:
    """Un tokenizador que le da un entero a cada carácter distinto."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # sorted() da el mismo orden en todas las corridas. Un set() pelado no,
        # y un modelo guardado tiene adentro identificadores, no caracteres.
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def train_val_split(
    data: torch.Tensor,
    val_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cortá la secuencia en una parte de entrenamiento y una de validación."""
    cut = int(len(data) * (1.0 - val_fraction))
    return data[:cut], data[cut:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tomá un batch de bloques al azar, y los mismos corridos una posición."""
    # y lee hasta la posición i + block_size, así que el offset válido más
    # grande es len(data) - block_size - 1. randint no incluye el valor high,
    # así que el argumento que le pasás es len(data) - block_size.
    high = len(data) - block_size
    offsets = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])
    return x, y
