"""Capítulo 2: código de referencia."""

from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Cuenta cuántas veces aparece cada par de tokens adyacentes."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Reemplaza cada aparición de `pair` por `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        # La segunda condición corta la lectura una posición antes del final.
        # La tercera es el caso del solapamiento: después de una coincidencia
        # el índice avanza dos posiciones, así que [1, 1, 1] da [9, 1].
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Un tokenizador BPE a nivel de bytes."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        """Aprende los merges a partir de un corpus."""
        if vocab_size < 256:
            raise ValueError(f"vocab_size tiene que ser 256 o más, no {vocab_size}")

        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break  # El texto es muy corto, no queda ningún par.
            top = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, top, new_id)
            self.merges[top] = new_id
            self.vocab[new_id] = self.vocab[top[0]] + self.vocab[top[1]]

    def encode(self, text: str) -> list[int]:
        """Convierte un string en una lista de identificadores de token."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # El índice de merge más bajo es el merge que train hizo primero. Un
            # par que no está en la tabla recibe infinito, así que min() nunca
            # lo elige.
            top = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if top not in self.merges:
                break  # Ningún par de esta lista está en la tabla de merges.
            ids = merge(ids, top, self.merges[top])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convierte una lista de identificadores de token de vuelta a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        # Una secuencia generada puede cortar un carácter multibyte, así que acá
        # puede saltar una excepción. U+FFFD es mejor respuesta que un crash.
        return raw.decode("utf-8", errors="replace")
