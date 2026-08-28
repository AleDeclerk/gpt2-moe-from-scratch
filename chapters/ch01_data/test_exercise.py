"""Tests del capítulo 1.

    uv run pytest chapters/ch01_data
"""

import pytest
import torch

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n"


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(1337)


# --- el tokenizador --------------------------------------------------------


def test_vocabulary_is_sorted_and_complete(target):
    tokenizer = target.CharTokenizer.from_text(TEXT)
    assert tokenizer.chars == sorted(set(TEXT))
    assert tokenizer.vocab_size == len(set(TEXT))


def test_encode_and_decode_are_inverse(target):
    tokenizer = target.CharTokenizer.from_text(TEXT)
    assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT


def test_encode_gives_valid_identifiers(target):
    tokenizer = target.CharTokenizer.from_text(TEXT)
    ids = tokenizer.encode(TEXT)
    assert len(ids) == len(TEXT), "un identificador por cada carácter"
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_vocabulary_does_not_depend_on_the_order_of_the_text(target):
    """Un modelo guardado tiene identificadores, así que el mapa no puede moverse."""
    first = target.CharTokenizer.from_text("abc")
    second = target.CharTokenizer.from_text("cba")
    assert first.stoi == second.stoi


# --- el split --------------------------------------------------------------


def test_split_keeps_every_element_in_order(target):
    data = torch.arange(100)
    train, val = target.train_val_split(data, val_fraction=0.1)
    assert torch.equal(torch.cat([train, val]), data), "el split tiene que ser contiguo"


def test_split_uses_the_end_for_validation(target):
    data = torch.arange(100)
    train, val = target.train_val_split(data, val_fraction=0.1)
    assert len(train) == 90
    assert len(val) == 10
    assert val[0].item() == 90


def test_split_accepts_another_fraction(target):
    data = torch.arange(1000)
    train, val = target.train_val_split(data, val_fraction=0.25)
    assert len(train) == 750
    assert len(val) == 250


# --- el batch --------------------------------------------------------------


def test_batch_has_the_correct_shape_and_type(target, generator):
    data = torch.arange(500)
    x, y = target.get_batch(data, batch_size=4, block_size=8, generator=generator)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_target_is_the_input_moved_one_position(target, generator):
    """Esta es la idea central del capítulo, así que el test lo dice explícito."""
    data = torch.arange(500)
    x, y = target.get_batch(data, batch_size=16, block_size=8, generator=generator)
    torch.testing.assert_close(y[:, :-1], x[:, 1:])


def test_batch_holds_real_windows_of_the_data(target, generator):
    data = torch.arange(500)
    x, y = target.get_batch(data, batch_size=16, block_size=8, generator=generator)
    for row_x, row_y in zip(x, y):
        start = row_x[0].item()
        torch.testing.assert_close(row_x, torch.arange(start, start + 8))
        torch.testing.assert_close(row_y, torch.arange(start + 1, start + 9))


def test_a_fixed_seed_gives_the_same_batch(target):
    data = torch.arange(500)
    first = target.get_batch(data, 4, 8, torch.Generator().manual_seed(42))
    second = target.get_batch(data, 4, 8, torch.Generator().manual_seed(42))
    assert torch.equal(first[0], second[0]), "pasale `generator` a torch.randint"
    assert torch.equal(first[1], second[1])


def test_the_last_block_is_reachable(target):
    """Con un único offset válido, el batch tiene que llegar al último token.

    Un offset sacado de len(data) es demasiado grande y rompe acá. Uno sacado
    de len(data) - block_size - 1 es demasiado chico y también rompe acá.
    """
    data = torch.arange(20)
    x, y = target.get_batch(data, batch_size=3, block_size=19)
    assert x[0, 0].item() == 0
    assert y[0, -1].item() == 19, "y tiene que llegar al último token de los datos"


def test_no_index_leaves_the_data(target, generator):
    data = torch.arange(64)
    x, y = target.get_batch(data, batch_size=256, block_size=16, generator=generator)
    assert x.max().item() < 64
    assert y.max().item() < 64
