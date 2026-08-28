"""Chapter 2 — Byte Pair Encoding.

Write the two functions and the class. Read README.md first.
Start with get_stats and merge, because train and encode both use them.
"""

from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how many times each pair of adjacent tokens is present.

    Args:
        ids: a list of token identifiers.

    Returns:
        A dict from a pair (a, b) to the number of times it is present.
        A list with less than two elements gives an empty dict.

    Example:
        get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}
    """
    raise NotImplementedError("get_stats")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`.

    Args:
        ids: a list of token identifiers.
        pair: the two adjacent identifiers to replace.
        new_id: the identifier of the new token.

    Returns:
        A new list. The input list does not change.

    Example:
        merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
        merge([1, 1, 1], (1, 1), 9) == [9, 1]

    Rules:
        Read the note about the overlap in README.md. The second example above
        is the case that a careless loop gets wrong.
    """
    raise NotImplementedError("merge")


class BPETokenizer:
    """A byte-level BPE tokenizer.

    Attributes:
        merges: a dict from a pair (a, b) to the identifier of the new token.
            The insertion order is the order of the merges, and encode needs it.
        vocab: a dict from an identifier to the bytes that it represents.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        """Learn the merges from a corpus.

        Args:
            text: the training corpus.
            vocab_size: the size of the final vocabulary. The value must be
                256 or more, because the 256 bytes are always present.

        Raises:
            ValueError: if vocab_size is less than 256.

        The method does vocab_size - 256 merges. If the text is too short and
        no pair is left, the method stops early.

        After the method, self.merges and self.vocab hold the result.
        """
        raise NotImplementedError("BPETokenizer.train")

    def encode(self, text: str) -> list[int]:
        """Map a string to a list of token identifiers.

        The merges must go in the same order as during the training. Of all
        the pairs that are present, take the pair with the lowest merge index.
        """
        raise NotImplementedError("BPETokenizer.encode")

    def decode(self, ids: list[int]) -> str:
        """Map a list of token identifiers back to a string.

        Use errors="replace" for the UTF-8 decode. README.md explains why.
        """
        raise NotImplementedError("BPETokenizer.decode")
