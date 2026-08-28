# Chapter 0 — Tensors, softmax, and gradients

## Why this chapter is first

Three operations appear in every later chapter. A softmax over the last
dimension turns attention scores into weights, and it also turns the final
logits into a probability for each token. A cross-entropy loss measures how
wrong those probabilities are. The chain rule moves that error back to every
parameter.

PyTorch has all three, and this course uses the PyTorch version after this
chapter. You write them one time here, because a bug in a later chapter is
easier to find when you know what these operations do.

## 1. A stable softmax

The definition is simple. For one row of numbers `z`:

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

The direct translation of that formula into code fails. With `z_i = 1000`,
`exp(1000)` is larger than the largest float, so the result is `inf`. Then
`inf / inf` gives `nan`, and the `nan` moves through the whole model.

The correction uses a property of the softmax. Subtract any constant `c` from
every element of `z`, and the result does not change:

```
exp(z_i - c) / sum_j exp(z_j - c) = [exp(z_i) exp(-c)] / [exp(-c) sum_j exp(z_j)]
```

The factor `exp(-c)` is present in the numerator and in the denominator, so it
cancels. The useful choice is `c = max(z)`. After the subtraction the largest
element is `0`, so `exp` returns a number between `0` and `1`. No overflow is
possible.

This is not a detail of style. GPT-2 divides attention scores by the square
root of the head dimension for a related reason, and Chapter 4 explains that
one.

**Shapes.** Your function receives a 2-D tensor with shape `(N, C)` and
returns a tensor with the same shape. Each row must have a sum of `1`. Watch
the `keepdim` argument of `max` and `sum`, because a reduction without
`keepdim` removes the dimension and breaks the broadcast.

## 2. Cross entropy

The model gives a score, called a logit, to each of the `C` tokens in the
vocabulary. The correct next token has index `t`. The loss is the negative
logarithm of the probability that the model gives to `t`:

```
loss = -log(softmax(z)_t)
```

A perfect model gives probability `1` to the correct token, and `-log(1)` is
`0`. A model that gives probability `0.001` receives a loss of about `6.9`.

There is a trap here as well. The expression `log(softmax(z))` calculates the
exponential and then the logarithm, and the two operations lose precision. The
algebra removes the round trip:

```
log_softmax(z)_i = z_i - max(z) - log(sum_j exp(z_j - max(z)))
```

Your function takes logits with shape `(N, C)` and targets with shape `(N,)`,
and returns one scalar: the mean loss over the `N` rows.

**A number to remember.** An untrained model gives about the same probability
to every token, so the loss is about `log(C)`. With a vocabulary of 65
characters that value is about `4.17`. Chapter 3 uses this number as the first
test of the first model. A training run that starts far above `log(C)` has a
bug in the initialization.

## 3. The backward pass of a linear layer

A linear layer is `y = x @ W + b`. During training, PyTorch gives you the
gradient of the loss for the output, `dL/dy`. You need three more gradients.

The rule for a matrix product is short:

```
dL/dx = dL/dy @ W.T
dL/dW = x.T @ dL/dy
dL/db = dL/dy summed over the batch dimension
```

Shape analysis is enough to remember these. With `x` of shape `(N, in)`, `W`
of shape `(in, out)`, and `dL/dy` of shape `(N, out)`, only one arrangement of
each product gives the correct shape. The bias gradient is a sum. The same
`b` is added to all `N` rows, so all `N` rows send an error to it.

Your function returns the three gradients as a tuple. The test compares them
against `loss.backward()` from PyTorch. Equal numbers mean that your
derivation is correct.

## Your task

1. Open `exercise.py`.
2. Write the three functions. Do not use `torch.softmax`, `torch.log_softmax`,
   `torch.nn.functional.cross_entropy`, or `backward`.
3. Run the tests.

```bash
uv run pytest chapters/ch00_tensors
```

4. After the tests pass, promote the code.

```bash
uv run python scripts/promote.py ch00
```

## Questions to answer for yourself

- The test `test_softmax_is_stable` uses logits of `1000`. Remove the
  subtraction of the maximum and run it. What is the output?
- Why does `dL/db` need a sum, but `dL/dW` needs a matrix product?
- The mean loss divides by `N`. Which gradient changes if the loss uses a sum
  instead, and by how much?
