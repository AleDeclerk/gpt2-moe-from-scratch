# Chapter 2 — Byte Pair Encoding

## Why not characters

The tokenizer of Chapter 1 has two problems, and both come from the same
decision. It builds the vocabulary from the characters of one corpus.

The first problem is coverage. A character that is not present in Tiny
Shakespeare has no identifier, so `encode` raises a KeyError. The letter `ñ`,
an emoji, and a Chinese character all fail.

The second problem is length. One character gives one token, so the model
needs a long context for a short text. Attention has a cost that grows with
the square of the sequence length, so length is expensive.

BPE solves both. GPT-2 uses it, and the vocabulary has 50257 tokens.

## 1. Bytes as the base vocabulary

BPE does not start from characters. It starts from bytes.

Every text has a UTF-8 representation, and every UTF-8 byte is a number
between 0 and 255. So a base vocabulary of 256 tokens can represent any text
in any language, with no unknown token. This property is the reason for the
choice.

```python
"hola".encode("utf-8")   # b'hola'        -> [104, 111, 108, 97]
"ñ".encode("utf-8")      # b'\xc3\xb1'    -> [195, 177]
```

The second line shows the cost. One character became two tokens. BPE gets that
cost back with the merges.

## 2. The training algorithm

The algorithm has three steps, and it repeats them:

1. Count how many times each pair of adjacent tokens is present.
2. Take the most frequent pair, and give it a new identifier.
3. Replace every occurrence of that pair with the new identifier.

An example with the word `aaabdaaabac`, as bytes:

```
start        a a a b d a a a b a c
most frequent pair: (a, a), 4 times.  Z = aa
after merge  Z a b d Z a b a c
most frequent pair: (Z, a), 2 times.  Y = Za
after merge  Y b d Y b a c
most frequent pair: (Y, b), 2 times.  X = Yb
after merge  X d X a c
```

Eleven tokens became five. The vocabulary grew from 4 to 7. This is the trade
of BPE, and the argument `vocab_size` controls it.

**Watch the overlap.** In `[1, 1, 1]` the pair `(1, 1)` is present two times,
but a merge can replace only one of them. The result is `[Z, 1]`, not `[Z, Z]`
and not `[Z]`. A loop with an explicit index handles this. A call to
`list.replace` does not.

## 3. Encode needs the order of the merges

The merges are not a set. They are an ordered list, and `encode` must apply
them in the same order as `train`.

The reason is that later merges are built from earlier merges. The token `X`
in the example above means `aaab`, but only because `Y` and `Z` exist already.
An `encode` that applies `X` first gets a different result, and the model
never saw those identifiers during training.

So `encode` repeats one step: of all the pairs that are present in the current
list, find the pair with the lowest merge index, and apply it. Stop when no
pair of the list is in the merge table.

## 4. What decode must survive

`decode` concatenates the bytes of each token, and then decodes UTF-8. A
random list of identifiers can give a byte sequence that is not valid UTF-8,
because a multi-byte character can be cut in the middle. During generation the
model can produce exactly that.

The standard answer is `errors="replace"`, which puts the character `U+FFFD`
in place of the invalid bytes. The alternative, an exception, stops the
generation of the model for a reason that is not interesting.

## Your task

1. Open `exercise.py`.
2. Write `get_stats`, `merge`, and the three methods of `BPETokenizer`.
3. Run the tests.

```bash
uv run pytest chapters/ch02_bpe
```

4. Promote the code.

```bash
uv run python scripts/promote.py ch02
```

The two functions come first, because `train` and `encode` both use them. Get
them green before you start the class.

## Questions to answer for yourself

- Train the tokenizer on Tiny Shakespeare with `vocab_size = 512`, and measure
  the compression. The test `test_compression_ratio` prints the number.
- The real GPT-2 tokenizer splits the text with a regular expression before
  BPE, so a merge never crosses a word boundary. What breaks without that
  step? Look at what `" the"`, `" the."`, and `" the,"` become.
- The vocabulary of GPT-2 has 50257 tokens. The number is 256 bytes, plus
  50000 merges, plus one. What is the last one, and why does a model need it?
