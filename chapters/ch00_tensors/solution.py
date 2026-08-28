"""Capítulo 0: código de referencia.

Leé este archivo después de intentarlo en serio, o cuando decidas seguir.
"""

import torch


def softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Calcula un softmax numéricamente estable en cada fila."""
    # keepdim=True mantiene el shape en (N, 1), así el broadcast contra (N, C)
    # sale bien. Sin eso el shape queda (N,) y la resta falla, o peor, hace el
    # broadcast sobre la dimensión equivocada.
    maximum = logits.max(dim=-1, keepdim=True).values
    shifted = logits - maximum
    exponential = shifted.exp()
    return exponential / exponential.sum(dim=-1, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calcula el loss de cross entropy promedio sobre un batch."""
    maximum = logits.max(dim=-1, keepdim=True).values
    shifted = logits - maximum
    # log(sum(exp(shifted))) es el log-sum-exp. La resta de arriba lo vuelve
    # seguro, porque el valor más grande de `shifted` es 0.
    log_denominator = shifted.exp().sum(dim=-1, keepdim=True).log()
    log_probabilities = shifted - log_denominator

    # gather() elige, para cada fila n, la columna targets[n].
    rows = log_probabilities.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -rows.mean()


def linear_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula los gradientes de una capa lineal y = x @ w + b."""
    grad_x = grad_out @ w.T
    grad_w = x.T @ grad_out
    # Cada fila del batch suma el mismo b, así que cada fila le manda su error.
    grad_b = grad_out.sum(dim=0)
    return grad_x, grad_w, grad_b
