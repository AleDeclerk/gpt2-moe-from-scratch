"""Chapter 1 — reference code."""

from __future__ import annotations

import torch


class CharTokenizer:
    """A tokenizer with one integer for each different character."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        # sorted() gives the same order on every run. A plain set() does not,
        # and a saved model holds identifiers, not characters.
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
    """Cut the sequence into a training part and a validation part."""
    cut = int(len(data) * (1.0 - val_fraction))
    return data[:cut], data[cut:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a batch of random blocks, and the same blocks moved one position."""
    # y reads position i + block_size, so the largest valid offset is
    # len(data) - block_size - 1. randint excludes the high value, so the
    # argument is len(data) - block_size.
    high = len(data) - block_size
    offsets = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])
    return x, y
