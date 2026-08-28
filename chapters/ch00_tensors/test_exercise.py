"""Tests for Chapter 0.

Green means your code agrees with PyTorch. Run them with:

    uv run pytest chapters/ch00_tensors
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def generator():
    """A generator with a fixed seed, so every run gives the same numbers."""
    return torch.Generator().manual_seed(1337)


# --- softmax ---------------------------------------------------------------


def test_softmax_matches_pytorch(target, generator):
    logits = torch.randn(8, 16, generator=generator)
    got = target.softmax_rows(logits)
    want = F.softmax(logits, dim=-1)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want)


def test_softmax_rows_sum_to_one(target, generator):
    logits = torch.randn(5, 11, generator=generator) * 7.0
    got = target.softmax_rows(logits)
    torch.testing.assert_close(got.sum(dim=-1), torch.ones(5))


def test_softmax_is_stable(target):
    """Large logits give inf without the subtraction of the maximum."""
    logits = torch.tensor([[1000.0, 1000.0, 1001.0], [-1000.0, -1000.0, -999.0]])
    got = target.softmax_rows(logits)
    assert torch.isfinite(got).all(), "the result has inf or nan, so subtract the row maximum"
    torch.testing.assert_close(got, F.softmax(logits, dim=-1))


# --- cross entropy ---------------------------------------------------------


def test_cross_entropy_matches_pytorch(target, generator):
    logits = torch.randn(32, 65, generator=generator)
    targets = torch.randint(0, 65, (32,), generator=generator)
    got = target.cross_entropy(logits, targets)
    want = F.cross_entropy(logits, targets)
    assert got.shape == (), "the loss must be a scalar, so use a mean over the batch"
    torch.testing.assert_close(got, want)


def test_cross_entropy_of_a_uniform_model(target):
    """Equal logits give a loss of log(C). Chapter 3 uses this value."""
    vocabulary = 65
    logits = torch.zeros(4, vocabulary)
    targets = torch.tensor([0, 1, 2, 3])
    got = target.cross_entropy(logits, targets)
    torch.testing.assert_close(got, torch.tensor(vocabulary).float().log())


def test_cross_entropy_is_stable(target):
    logits = torch.tensor([[0.0, 0.0, 800.0], [800.0, 0.0, 0.0]])
    targets = torch.tensor([0, 0])
    got = target.cross_entropy(logits, targets)
    assert torch.isfinite(got), "the result is inf or nan, so do not call log() after exp()"
    torch.testing.assert_close(got, F.cross_entropy(logits, targets))


# --- backward pass ---------------------------------------------------------


def test_linear_backward_matches_autograd(target, generator):
    x = torch.randn(6, 4, generator=generator, requires_grad=True)
    w = torch.randn(4, 3, generator=generator, requires_grad=True)
    b = torch.randn(3, generator=generator, requires_grad=True)
    grad_out = torch.randn(6, 3, generator=generator)

    # The reference: let PyTorch do the work.
    y = x @ w + b
    y.backward(grad_out)

    got_x, got_w, got_b = target.linear_backward(x.detach(), w.detach(), grad_out)

    assert got_x.shape == x.shape, f"grad_x must have shape {tuple(x.shape)}"
    assert got_w.shape == w.shape, f"grad_w must have shape {tuple(w.shape)}"
    assert got_b.shape == b.shape, f"grad_b must have shape {tuple(b.shape)}"
    torch.testing.assert_close(got_x, x.grad)
    torch.testing.assert_close(got_w, w.grad)
    torch.testing.assert_close(got_b, b.grad)


def test_linear_backward_with_one_row(target, generator):
    """A batch of one row still needs a sum for the bias, not a copy."""
    x = torch.randn(1, 5, generator=generator, requires_grad=True)
    w = torch.randn(5, 2, generator=generator, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    grad_out = torch.randn(1, 2, generator=generator)

    (x @ w + b).backward(grad_out)
    got_x, got_w, got_b = target.linear_backward(x.detach(), w.detach(), grad_out)

    torch.testing.assert_close(got_x, x.grad)
    torch.testing.assert_close(got_w, w.grad)
    torch.testing.assert_close(got_b, b.grad)
