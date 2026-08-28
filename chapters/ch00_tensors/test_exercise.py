"""Tests del capítulo 0.

Verde quiere decir que tu código coincide con PyTorch. Corré los tests así:

    uv run pytest chapters/ch00_tensors
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def generator():
    """Un generador con seed fija, así cada corrida da los mismos números."""
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
    """Si no restás el máximo, los logits grandes dan inf."""
    logits = torch.tensor([[1000.0, 1000.0, 1001.0], [-1000.0, -1000.0, -999.0]])
    got = target.softmax_rows(logits)
    assert torch.isfinite(got).all(), "el resultado tiene inf o nan: restá el máximo de la fila"
    torch.testing.assert_close(got, F.softmax(logits, dim=-1))


# --- cross entropy ---------------------------------------------------------


def test_cross_entropy_matches_pytorch(target, generator):
    logits = torch.randn(32, 65, generator=generator)
    targets = torch.randint(0, 65, (32,), generator=generator)
    got = target.cross_entropy(logits, targets)
    want = F.cross_entropy(logits, targets)
    assert got.shape == (), "el loss tiene que ser un escalar: promediá sobre el batch"
    torch.testing.assert_close(got, want)


def test_cross_entropy_of_a_uniform_model(target):
    """Con logits iguales el loss da log(C). El capítulo 3 usa este valor."""
    vocabulary = 65
    logits = torch.zeros(4, vocabulary)
    targets = torch.tensor([0, 1, 2, 3])
    got = target.cross_entropy(logits, targets)
    torch.testing.assert_close(got, torch.tensor(vocabulary).float().log())


def test_cross_entropy_is_stable(target):
    logits = torch.tensor([[0.0, 0.0, 800.0], [800.0, 0.0, 0.0]])
    targets = torch.tensor([0, 0])
    got = target.cross_entropy(logits, targets)
    assert torch.isfinite(got), "el resultado es inf o nan: no llames a log() después de exp()"
    torch.testing.assert_close(got, F.cross_entropy(logits, targets))


# --- el backward -----------------------------------------------------------


def test_linear_backward_matches_autograd(target, generator):
    x = torch.randn(6, 4, generator=generator, requires_grad=True)
    w = torch.randn(4, 3, generator=generator, requires_grad=True)
    b = torch.randn(3, generator=generator, requires_grad=True)
    grad_out = torch.randn(6, 3, generator=generator)

    # La referencia: que el trabajo lo haga PyTorch.
    y = x @ w + b
    y.backward(grad_out)

    got_x, got_w, got_b = target.linear_backward(x.detach(), w.detach(), grad_out)

    assert got_x.shape == x.shape, f"grad_x tiene que tener shape {tuple(x.shape)}"
    assert got_w.shape == w.shape, f"grad_w tiene que tener shape {tuple(w.shape)}"
    assert got_b.shape == b.shape, f"grad_b tiene que tener shape {tuple(b.shape)}"
    torch.testing.assert_close(got_x, x.grad)
    torch.testing.assert_close(got_w, w.grad)
    torch.testing.assert_close(got_b, b.grad)


def test_linear_backward_with_one_row(target, generator):
    """Un batch de una fila igual necesita una suma para el bias, no una copia."""
    x = torch.randn(1, 5, generator=generator, requires_grad=True)
    w = torch.randn(5, 2, generator=generator, requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    grad_out = torch.randn(1, 2, generator=generator)

    (x @ w + b).backward(grad_out)
    got_x, got_w, got_b = target.linear_backward(x.detach(), w.detach(), grad_out)

    torch.testing.assert_close(got_x, x.grad)
    torch.testing.assert_close(got_w, w.grad)
    torch.testing.assert_close(got_b, b.grad)
