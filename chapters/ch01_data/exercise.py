"""Chapter 1 — the data pipeline.

Write the class and the two functions. Read README.md first.
"""

from __future__ import annotations

import torch


class CharTokenizer:
    """A tokenizer with one integer for each different character.

    Attributes:
        chars: the sorted list of the characters in the vocabulary.
        stoi: a dict that maps a character to its integer.
        itos: a dict that maps an integer to its character.
    """

    def __init__(self, chars: list[str]) -> None:
        """Build the two maps from a sorted list of characters."""
        raise NotImplementedError("CharTokenizer.__init__")

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer from the characters that are present in `text`.

        The order must be stable between runs, so sort the characters.
        """
        raise NotImplementedError("CharTokenizer.from_text")

    @property
    def vocab_size(self) -> int:
        """Get the number of different tokens."""
        raise NotImplementedError("CharTokenizer.vocab_size")

    def encode(self, text: str) -> list[int]:
        """Map a string to a list of integers."""
        raise NotImplementedError("CharTokenizer.encode")

    def decode(self, ids: list[int]) -> str:
        """Map a list of integers back to a string."""
        raise NotImplementedError("CharTokenizer.decode")


def train_val_split(
    data: torch.Tensor,
    val_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut the sequence into a training part and a validation part.

    Args:
        data: a 1-D tensor of token identifiers.
        val_fraction: the part of the data for validation, between 0 and 1.

    Returns:
        A tuple (train, val). The two parts are contiguous, and `val` is the
        end of the sequence. Together they hold every element of `data`.

    Rules:
        Do not shuffle. README.md explains why a random split gives a
        validation loss that is too good.
    """
    raise NotImplementedError("train_val_split")


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a batch of random blocks, and the same blocks moved one position.

    Args:
        data: a 1-D tensor of token identifiers.
        batch_size: the number of blocks, B.
        block_size: the number of tokens in each block, T.
        generator: a torch.Generator for the random offsets, or None.

    Returns:
        A tuple (x, y). Both tensors have shape (B, T) and dtype int64.
        y[b, t] is the token that comes after x[b, t].

    Rules:
        Use torch.randint with the `generator` argument, so a fixed seed gives
        the same batch every time.
        Watch the largest valid offset. y reads one position beyond x.
    """
    raise NotImplementedError("get_batch")
