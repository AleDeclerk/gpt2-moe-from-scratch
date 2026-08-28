# Design — gpt2-moe-from-scratch

Date: 2026-08-28. Status: approved, in construction.

## Goal

One repository that teaches GPT-2 and Mixture of Experts, with code that the
reader writes. The reader is the owner of the repository, and the target is a
solid understanding of every part, not a fast result.

## Decisions

| Decision | Choice | Reason |
|---|---|---|
| Exercise format | Python modules with pytest | Green is not an opinion. A notebook has no clear pass condition, and git handles a `.py` file much better |
| Start point | Everything from zero, tokenizer included | The request says "review the concepts". A given base removes exactly the review |
| MoE depth | Top-k, load balance, capacity, diagnostics | A router alone collapses to one expert. The other parts are the answer to that problem |
| Chapter coupling | Promotion | See below |
| Language | English, ASD-STE100 rules | Global rule for everything that goes to GitHub |

## The promotion model

Three models were possible for the relation between chapters:

1. **Self-contained.** Each chapter holds a copy of the code of the chapters
   before it. No coupling, but the reader writes GPT-2 twelve times.
2. **Shared package, given.** The package `gpt2moe/` arrives complete, and the
   chapters import from it. No duplication, but the solution is visible from
   day one.
3. **Promotion (the choice).** The reader writes `exercise.py`. After the
   tests pass, `scripts/promote.py` copies the file into `gpt2moe/`. The next
   chapter imports from the package, so it imports the code of the reader.

The third model has no duplication and no early solution, and the package
gives a visible measure of progress. The cost is one command for each chapter.

`--from-solution` promotes the reference code without a test run. This is the
exit for a chapter that blocks the reader.

## Test target selection

The environment variable `MOE_TARGET` decides between `exercise` and
`solution`. The fixture `target` in `conftest.py` loads the file by path, with
a unique module name for each chapter. The unique name is necessary, because
every chapter has a file with the same two names.

This is the same pattern as `PREP_TARGET` in `python-intermediate-prep`.

## Model scale

6 layers, 6 heads, 384 dimensions, a context of 256 tokens, about 10 million
parameters. A training run takes minutes on MPS. The MoE variant of Chapter 12
uses 8 experts with top-2. That gives about 4 times the total parameters, with
almost the same number of active parameters.

## Acceptance criteria

| Chapter | Condition |
|---|---|
| ch05 | The vectorized attention agrees with the loop version, `allclose` |
| ch07 | The model with the real `gpt2` weights gives the reference logits, atol 1e-4 |
| ch08 | The dense model goes under the validation loss target |
| ch11 | The vectorized MoE agrees with the naive MoE, `allclose` |
| ch12 | The MoE model gets a lower validation loss than the dense model, at equal active parameters |

The criterion of Chapter 7 is the strongest one. A model that produces the
logits of the real GPT-2 from the real weights is GPT-2, not something similar.

## State

Complete: the scaffolding, ch00, ch01, ch02. 39 tests.
Next: ch03 to ch08, then the MoE part.
