"""Chapter 0 — tensors, softmax, and gradients.

Write the three functions. Read README.md first.
"""

import torch


def softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Apply a numerically stable softmax to each row.

    Args:
        logits: a tensor with shape (N, C).

    Returns:
        A tensor with shape (N, C). Each row has a sum of 1.

    Rules:
        Do not use torch.softmax or torch.nn.functional.softmax.
        Subtract the maximum of each row before the exponential.
    """
    raise NotImplementedError("softmax_rows")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Get the mean cross-entropy loss over a batch.

    Args:
        logits: a tensor with shape (N, C).
        targets: a tensor of integers with shape (N,). Each value is a column
            index of the correct class, between 0 and C - 1.

    Returns:
        A scalar tensor: the mean loss over the N rows.

    Rules:
        Do not use torch.nn.functional.cross_entropy or torch.log_softmax.
        Calculate the log-softmax directly, without a call to log() after exp().
    """
    raise NotImplementedError("cross_entropy")


def linear_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the gradients of a linear layer y = x @ w + b.

    Args:
        x: the input, with shape (N, in_features).
        w: the weight, with shape (in_features, out_features).
        grad_out: the gradient of the loss for y, with shape (N, out_features).

    Returns:
        A tuple (grad_x, grad_w, grad_b) with the shapes of x, w, and b.
        The bias b has shape (out_features,).

    Rules:
        Do not call backward(). Write the three expressions yourself.
        The function does not need the value of b, because the gradient for a
        sum does not depend on the operands.
    """
    raise NotImplementedError("linear_backward")
