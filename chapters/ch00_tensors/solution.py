"""Chapter 0 — reference code.

Read this file after an honest try, or when you decide to continue.
"""

import torch


def softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Apply a numerically stable softmax to each row."""
    # keepdim=True holds the shape at (N, 1), so the broadcast against (N, C)
    # is correct. Without it the shape is (N,) and the subtraction fails, or
    # worse, it broadcasts over the wrong dimension.
    maximum = logits.max(dim=-1, keepdim=True).values
    shifted = logits - maximum
    exponential = shifted.exp()
    return exponential / exponential.sum(dim=-1, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Get the mean cross-entropy loss over a batch."""
    maximum = logits.max(dim=-1, keepdim=True).values
    shifted = logits - maximum
    # log(sum(exp(shifted))) is the log-sum-exp. The subtraction above makes it
    # safe, because the largest value of `shifted` is 0.
    log_denominator = shifted.exp().sum(dim=-1, keepdim=True).log()
    log_probabilities = shifted - log_denominator

    # gather() selects, for each row n, the column targets[n].
    rows = log_probabilities.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -rows.mean()


def linear_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the gradients of a linear layer y = x @ w + b."""
    grad_x = grad_out @ w.T
    grad_w = x.T @ grad_out
    # Each row of the batch adds the same b, so each row sends an error to it.
    grad_b = grad_out.sum(dim=0)
    return grad_x, grad_w, grad_b
