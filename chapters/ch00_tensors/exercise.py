"""Capítulo 0: tensores, softmax y gradientes.

Escribí las tres funciones. Leé primero README.md.
"""

import torch


def softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Calcula un softmax numéricamente estable en cada fila.

    Argumentos:
        logits: un tensor de shape (N, C).

    Devuelve:
        Un tensor de shape (N, C). Cada fila suma 1.

    Reglas:
        No uses torch.softmax ni torch.nn.functional.softmax.
        Restá el máximo de cada fila antes de la exponencial.
    """
    raise NotImplementedError("softmax_rows")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calcula el loss de cross entropy promedio sobre un batch.

    Argumentos:
        logits: un tensor de shape (N, C).
        targets: un tensor de enteros de shape (N,). Cada valor es el índice de
            columna de la clase correcta, entre 0 y C - 1.

    Devuelve:
        Un tensor escalar: el loss promedio sobre las N filas.

    Reglas:
        No uses torch.nn.functional.cross_entropy ni torch.log_softmax.
        Calculá el log-softmax directo, sin llamar a log() después de exp().
    """
    raise NotImplementedError("cross_entropy")


def linear_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula los gradientes de una capa lineal y = x @ w + b.

    Argumentos:
        x: la entrada, de shape (N, in_features).
        w: los pesos, de shape (in_features, out_features).
        grad_out: el gradiente del loss respecto de y, de shape
            (N, out_features).

    Devuelve:
        Una tupla (grad_x, grad_w, grad_b) con los shapes de x, w y b.
        El bias b tiene shape (out_features,).

    Reglas:
        No llames a backward(). Escribí vos las tres expresiones.
        La función no necesita el valor de b, porque el gradiente de una suma
        no depende de los operandos.
    """
    raise NotImplementedError("linear_backward")
