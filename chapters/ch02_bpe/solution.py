"""Chapter 2 — reference code."""

from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how many times each pair of adjacent tokens is present."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        # The second condition stops the read one position before the end.
        # The third condition is the overlap case: after a match the index
        # moves two positions, so [1, 1, 1] gives [9, 1].
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        """Learn the merges from a corpus."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be 256 or more, not {vocab_size}")

        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break  # The text is too short, so no pair is left.
            top = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, top, new_id)
            self.merges[top] = new_id
            self.vocab[new_id] = self.vocab[top[0]] + self.vocab[top[1]]

    def encode(self, text: str) -> list[int]:
        """Map a string to a list of token identifiers."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # The lowest merge index is the merge that train did first. A pair
            # that is not in the table gets infinity, so min() never takes it.
            top = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if top not in self.merges:
                break  # No pair of this list is in the merge table.
            ids = merge(ids, top, self.merges[top])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Map a list of token identifiers back to a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        # A generated sequence can cut a multi-byte character, so an exception
        # is possible here. U+FFFD is a better answer than a crash.
        return raw.decode("utf-8", errors="replace")
