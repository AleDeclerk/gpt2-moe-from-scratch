"""Capítulo 2: Byte Pair Encoding.

Escribí las dos funciones y la clase. Leé primero README.md.
Arrancá por get_stats y merge, porque train y encode usan las dos.
"""

from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Cuenta cuántas veces aparece cada par de tokens adyacentes.

    Argumentos:
        ids: una lista de identificadores de token.

    Devuelve:
        Un dict del par (a, b) a la cantidad de veces que aparece.
        Una lista con menos de dos elementos da un dict vacío.

    Ejemplo:
        get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}
    """
    raise NotImplementedError("get_stats")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Reemplaza cada aparición de `pair` por `new_id`.

    Argumentos:
        ids: una lista de identificadores de token.
        pair: los dos identificadores adyacentes que hay que reemplazar.
        new_id: el identificador del token nuevo.

    Devuelve:
        Una lista nueva. La lista de entrada no cambia.

    Ejemplo:
        merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
        merge([1, 1, 1], (1, 1), 9) == [9, 1]

    Reglas:
        Leé la nota sobre el solapamiento en README.md. El segundo ejemplo de
        acá arriba es el caso que un loop descuidado resuelve mal.
    """
    raise NotImplementedError("merge")


class BPETokenizer:
    """Un tokenizador BPE a nivel de bytes.

    Atributos:
        merges: un dict del par (a, b) al identificador del token nuevo.
            El orden de inserción es el orden de los merges, y encode lo
            necesita.
        vocab: un dict del identificador a los bytes que representa.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        """Aprende los merges a partir de un corpus.

        Argumentos:
            text: el corpus de entrenamiento.
            vocab_size: el tamaño del vocabulario final. Tiene que ser 256 o
                más, porque los 256 bytes están siempre.

        Lanza:
            ValueError: si vocab_size es menor que 256.

        El método hace vocab_size - 256 merges. Si el texto es muy corto y no
        queda ningún par, corta antes.

        Cuando termina, self.merges y self.vocab tienen el resultado.
        """
        raise NotImplementedError("BPETokenizer.train")

    def encode(self, text: str) -> list[int]:
        """Convierte un string en una lista de identificadores de token.

        Los merges tienen que ir en el mismo orden que en el entrenamiento. De
        todos los pares que aparecen, tomá el de índice de merge más bajo.
        """
        raise NotImplementedError("BPETokenizer.encode")

    def decode(self, ids: list[int]) -> str:
        """Convierte una lista de identificadores de token de vuelta a string.

        Usá errors="replace" en el decode UTF-8. README.md explica por qué.
        """
        raise NotImplementedError("BPETokenizer.decode")
