# GPT-2 with Mixture of Experts, from scratch

A course in the form of a repository. You write the code, and the tests tell
you if the code is correct.

The course builds two language models. The first one is GPT-2, the dense
architecture from 2019. The second one replaces each feed-forward block with a
Mixture of Experts layer. Mixtral, DeepSeek, and most large models of today use
that architecture. At the end you compare the two, with the same
number of active parameters.

Nothing here is a black box. You write the tokenizer, the attention, the
transformer block, the training loop, the router, and the load-balance loss.

## How the course works

Each chapter is one directory in `chapters/`, with four files:

| File | What it is |
|---|---|
| `README.md` | The theory, and the description of your task |
| `exercise.py` | Your code. Each function starts with `raise NotImplementedError` |
| `test_exercise.py` | The tests. Green means that your code is correct |
| `solution.py` | The reference code, for the moment when you are blocked |

You build the package `gpt2moe/` yourself. It is almost empty now. After the
tests of a chapter pass, promote your code:

```bash
uv run python scripts/promote.py ch00
```

The command copies your `exercise.py` into `gpt2moe/`, but only if the tests
pass. The next chapter imports from the package, so it imports your own code.
By Chapter 8 you train a model that is your work from the tokenizer to the
optimizer.

## The website

<https://gpt2-moe-from-scratch.vercel.app>

The site shows the theory of each chapter and the state of the course. It
reads the README of each chapter, so the text has one home only.

The progress is a measurement, not a declaration:

1. You make the tests of a chapter pass.
2. You push.
3. A GitHub Action runs pytest for every chapter, and writes the result of
   each test to `progress.json`.
4. Vercel builds the site again from that file.

The site holds no state of its own, so it cannot show a green chapter while
the tests fail. To see the numbers before a push, run the measurement:

```bash
uv run python scripts/sync_progress.py
```

## Install

1. Install `uv`, if the computer does not have it.
2. Run `uv sync`. The command creates the environment and installs PyTorch.
3. Run `uv run python scripts/get_data.py` to get the corpus.
4. Run the first tests. They fail, and that is the correct start.

```bash
uv run pytest chapters/ch00_tensors
```

## Commands

| Command | What it does |
|---|---|
| `uv run pytest chapters/ch00_tensors` | Test one chapter |
| `uv run pytest` | Test every chapter |
| `uv run pytest -m "not slow"` | Skip the tests that need a download |
| `MOE_TARGET=solution uv run pytest` | Test the reference code, not yours |
| `uv run python scripts/promote.py ch00` | Move your validated code into `gpt2moe/` |
| `uv run python scripts/promote.py ch00 --from-solution` | Move the reference code, to continue |

## The chapters

### Part 1 — The base

| Chapter | Subject | What you write |
|---|---|---|
| `ch00_tensors` | Tensors, softmax, gradients | A stable softmax, cross entropy, a backward pass by hand |
| `ch01_data` | The data pipeline | A character tokenizer, the split, the batch sampler |
| `ch02_bpe` | Byte Pair Encoding | The BPE algorithm, and a comparison against `tiktoken` |

### Part 2 — The transformer

| Chapter | Subject | What you write |
|---|---|---|
| `ch03_embeddings` | Embeddings and a baseline | A bigram model, and the first sanity check at `log(V)` |
| `ch04_attention` | Self-attention | One head, the causal mask, the scale of `1/sqrt(d)` |
| `ch05_multihead` | Multi-head attention | The version with a loop, and the vectorized version |
| `ch06_block` | The transformer block | The feed-forward layer, GELU, pre-LN, the residual path |
| `ch07_gpt2` | The complete model | GPT-2, with weight tying and the original initialization |
| `ch08_training` | The training loop | AdamW, warmup, cosine schedule, sampling |

### Part 3 — Mixture of Experts

| Chapter | Subject | What you write |
|---|---|---|
| `ch09_moe` | The first MoE layer | The router, top-k selection, the weighted combination |
| `ch10_balance` | The collapse problem | Expert use metrics, the auxiliary loss, the router z-loss |
| `ch11_capacity` | Capacity and dispatch | The capacity factor, token drop, a vectorized dispatch |
| `ch12_compare` | Dense against sparse | The experiment, and the comparison at equal active parameters |
| `ch13_ablations` | Ablations | Top-1 against top-2, the number of experts, no auxiliary loss |

## How you know that the model is correct

Each part ends with a test that is difficult to pass by accident.

| Chapter | Green means |
|---|---|
| `ch05` | The vectorized attention agrees with the version that uses a loop |
| `ch07` | Your GPT-2, with the real weights of `gpt2`, gives the same logits as the reference (atol 1e-4). This proves that your model **is** GPT-2 |
| `ch08` | The dense model goes under the validation loss target |
| `ch11` | The vectorized MoE agrees with the naive MoE |
| `ch12` | The MoE model beats the dense model at equal active parameters |

## Hardware

The models are small on purpose: 6 layers, 6 heads, 384 dimensions, a context
of 256 tokens, and about 10 million parameters. A training run takes minutes
on the GPU of an Apple Silicon machine, through the MPS backend of PyTorch. A
CPU also works, with more patience.

One test in Chapter 7 downloads about 500 MB of weights from Hugging Face. It
carries the mark `slow`, so the rest of the course works without a network.

## Credits

The path from the bigram model to GPT-2 follows the structure of *Let's build
GPT* and `nanoGPT`, by Andrej Karpathy. The BPE chapter follows `minbpe`, from
the same author. The Mixture of Experts part uses three papers:

- Switch Transformer, Fedus et al., 2021
- ST-MoE, Zoph et al., 2022
- Mixtral of Experts, Jiang et al., 2024
