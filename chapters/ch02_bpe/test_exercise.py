"""Tests for Chapter 2.

    uv run pytest chapters/ch02_bpe
"""

from pathlib import Path

import pytest

CORPUS = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    "All:\nSpeak, speak.\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n"
) * 20

DATA = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"


# --- get_stats -------------------------------------------------------------


def test_get_stats_counts_adjacent_pairs(target):
    assert target.get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_get_stats_counts_an_overlap_two_times(target):
    assert target.get_stats([1, 1, 1]) == {(1, 1): 2}


def test_get_stats_of_a_short_list_is_empty(target):
    assert target.get_stats([]) == {}
    assert target.get_stats([7]) == {}


# --- merge -----------------------------------------------------------------


def test_merge_replaces_every_occurrence(target):
    assert target.merge([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]


def test_merge_handles_the_overlap(target):
    """[1, 1, 1] has the pair two times, but only one merge is possible."""
    assert target.merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert target.merge([1, 1, 1, 1], (1, 1), 9) == [9, 9]


def test_merge_keeps_the_end_of_the_list(target):
    assert target.merge([5, 1, 2], (1, 2), 9) == [5, 9]
    assert target.merge([1, 2, 5], (1, 2), 9) == [9, 5]
    assert target.merge([4, 4], (1, 2), 9) == [4, 4]


def test_merge_does_not_change_the_input(target):
    ids = [1, 2, 3]
    target.merge(ids, (1, 2), 9)
    assert ids == [1, 2, 3], "return a new list"


def test_merge_of_the_readme_example(target):
    ids = list(b"aaabdaaabac")
    a, b, c, d = ord("a"), ord("b"), ord("c"), ord("d")
    ids = target.merge(ids, (a, a), 256)
    assert ids == [256, a, b, d, 256, a, b, a, c]
    ids = target.merge(ids, (256, a), 257)
    assert ids == [257, b, d, 257, b, a, c]
    ids = target.merge(ids, (257, b), 258)
    assert ids == [258, d, 258, a, c]


# --- train -----------------------------------------------------------------


def test_train_gives_the_requested_vocabulary_size(target):
    tokenizer = target.BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300)
    assert len(tokenizer.vocab) == 300
    assert len(tokenizer.merges) == 44


def test_train_rejects_a_vocabulary_under_256(target):
    tokenizer = target.BPETokenizer()
    with pytest.raises(ValueError):
        tokenizer.train(CORPUS, vocab_size=100)


def test_train_stops_early_when_no_pair_is_left(target):
    """A one-character text has no pair, so the loop must stop, not crash."""
    tokenizer = target.BPETokenizer()
    tokenizer.train("a", vocab_size=300)
    assert len(tokenizer.merges) == 0


def test_vocabulary_holds_the_bytes_of_each_token(target):
    tokenizer = target.BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=280)
    for pair, new_id in tokenizer.merges.items():
        expected = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        assert tokenizer.vocab[new_id] == expected


# --- encode and decode -----------------------------------------------------


def test_encode_and_decode_are_inverse(target):
    tokenizer = target.BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300)
    text = "First Citizen: hear me speak."
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_of_text_that_is_not_in_the_corpus(target):
    """A byte vocabulary has no unknown token, so any text must survive."""
    tokenizer = target.BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300)
    for text in ["señor", "日本語", "emoji: 🤖", "", "\t\n\x00"]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_applies_the_merges_in_order(target):
    """A later merge is built from an earlier one, so the order is the result."""
    tokenizer = target.BPETokenizer()
    tokenizer.train("aaabdaaabac", vocab_size=259)
    a, b, c, d = ord("a"), ord("b"), ord("c"), ord("d")
    assert tokenizer.merges == {(a, a): 256, (256, a): 257, (257, b): 258}
    assert tokenizer.encode("aaabdaaabac") == [258, d, 258, a, c]


def test_encode_compresses_the_text(target):
    tokenizer = target.BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=400)
    raw = len(CORPUS.encode("utf-8"))
    tokens = len(tokenizer.encode(CORPUS))
    assert tokens < raw, "more tokens than bytes means that the merges do not apply"


def test_decode_survives_an_invalid_byte_sequence(target):
    """Generation can cut a multi-byte character, so decode must not raise."""
    tokenizer = target.BPETokenizer()
    tokenizer.train(CORPUS, vocab_size=300)
    assert tokenizer.decode([195]) == "�"


# --- the real corpus -------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not DATA.exists(), reason="run scripts/get_data.py first")
def test_compression_ratio(target, capsys):
    """Measure the trade of chapter 1 against chapter 2. This test prints."""
    text = DATA.read_text(encoding="utf-8")[:200_000]
    tokenizer = target.BPETokenizer()
    tokenizer.train(text, vocab_size=512)
    tokens = len(tokenizer.encode(text))
    with capsys.disabled():
        print(f"\n  characters: {len(text)}")
        print(f"  BPE tokens: {tokens}")
        print(f"  ratio:      {len(text) / tokens:.2f} characters for each token")
    assert len(text) / tokens > 1.5
