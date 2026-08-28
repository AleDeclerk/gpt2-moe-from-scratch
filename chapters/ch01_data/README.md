# Chapter 1 — The data pipeline

## What a language model learns

The task is one sentence long: given the text until now, give a probability to
each token that can come next. Everything else in this course is machinery for
that one question.

So the training data is not a table of inputs and labels. It is one long
sequence of tokens, and the label of every position is the token at the next
position. The dataset writes itself, and this is why the method is called
self-supervised.

## 1. A character tokenizer

A tokenizer maps text to integers. The simplest version uses one integer per
character. Take the set of characters in the corpus, sort them, and use the
position in that sorted list as the identifier.

Tiny Shakespeare has 65 different characters, so the vocabulary is 65 tokens.
This is a very small vocabulary. GPT-2 uses 50257 tokens, and Chapter 2 builds
that kind of tokenizer with the BPE algorithm.

The two sizes trade against each other:

| | Character tokens | BPE tokens |
|---|---|---|
| Vocabulary | 65 | 50257 |
| Tokens for the same text | many | about 4 times less |
| Embedding table | small | large |
| Context of 256 tokens holds | about 256 characters | about 1000 characters |

A small vocabulary gives long sequences, and attention has a cost that grows
with the square of the sequence length. A large vocabulary gives short
sequences, but a large embedding table and a large output layer.

**The sort matters.** Two runs must give the same identifier to the same
character, because a saved model holds the identifiers, not the characters.
`sorted(set(text))` is stable. `set(text)` alone is not.

## 2. The train and validation split

The course keeps the last 10 percent of the text for validation, and the split
is contiguous. A random split of positions is wrong here, and the reason is
specific to sequence data.

The training examples are windows over the text, and the windows overlap. A
random split puts the window at position 100 in the training set. It puts the
window at position 101 in the validation set. The two
windows share 255 of their 256 characters. The validation loss then measures
memory, not generalization, and it looks much better than the truth.

## 3. One batch, many examples

This is the part that surprises people. A block of 256 tokens is not one
training example. It is 256 training examples in one tensor.

Take the block `[F, i, r, s, t]`. The model sees this:

```
input                 target
[F]                -> i
[F, i]             -> r
[F, i, r]          -> s
[F, i, r, s]       -> t
```

The causal mask of Chapter 4 makes all four predictions in one forward pass.
Position `t` can read positions `0` up to `t`, and nothing after. So
the code does not build the four rows above. The code builds two tensors:

```
x = data[i     : i + T]        the block
y = data[i + 1 : i + T + 1]    the same block, moved one position
```

Then `y[t]` is the correct answer for the prefix that ends at `x[t]`. One
tensor of shape `(B, T)` holds `B * T` predictions. With `B = 32` and
`T = 256`, one step of training uses 8192 examples.

**The consequence for the offset.** The start position `i` must satisfy
`i + T + 1 <= len(data)`, because `y` reads one position beyond `x`. An offset
drawn from `len(data) - T` is off by one and fails on the last block.

## Your task

1. Open `exercise.py`.
2. Write `CharTokenizer`, `train_val_split`, and `get_batch`.
3. Run the tests.

```bash
uv run pytest chapters/ch01_data
```

4. Promote the code.

```bash
uv run python scripts/promote.py ch01
```

The tests use a small text of their own, so they do not need a download. To
get the real corpus for Chapter 8, run `uv run python scripts/get_data.py`.

## Questions to answer for yourself

- Tiny Shakespeare has 1115394 characters. How many tokens does the BPE
  tokenizer of GPT-2 need for the same text? Chapter 2 measures it.
- With `B = 32` and `T = 256`, how many bytes does one batch use as `int64`?
  And as `int32`?
- The validation split is the end of the text. Print the last 500 characters.
  Is that part of the corpus the same kind of text as the start?
