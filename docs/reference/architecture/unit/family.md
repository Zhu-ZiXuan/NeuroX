# Unit family

A unit realizes one exact-integer tensor operator by lowering it to a matrix-multiplication-shaped execution substrate and undoing every representation axis that lowering introduces.

## Operator law

Over its accepted integer value domain, a unit returns the result of its declared operator before requantization. Any deviation from exact integer arithmetic belongs to the execution substrate rather than to the lowering.

Every operator lowers through three seams: a program-time weight-to-matrix map, a call-time activation-to-planes map, and an aggregation map that removes exactly the axes introduced by the first two. A shape change introduced by lowering is undone there; input-leading axes pass through without reduction, reordering, or interpretation.

When the operator carries an integer bias, the bias is programmed with the weight and added in the int64 accumulation domain after substrate aggregation. It introduces no execution cycle or dynamic-energy event of its own.

## Shared conventions

A unit accepts integer weights and activations within its published value ranges and returns a pre-requantize integer tensor. Requantization to an activation grid is outside the unit model.

Geometric placement partitions the contraction dimension, assigns logical output blocks to balanced groups, and restores those blocks to logical output order. It changes only the matrix geometry and never decomposes a value.

## Governing laws

For a logical weight matrix $\mathbf{W}$ of shape $(N, K)$ and an activation matrix $\mathbf{X}$ of shape $(M, K)$,

$$\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}, \qquad Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

with both operands restricted to the unit's accepted integer value ranges. The operator-specific maps arrange the operands around this contraction and restore the declared output shape.

If the contraction dimension is partitioned, results from its partitions are accumulated. Output blocks are not reduced; they are restored to logical output order.

## Noise & non-idealities

Lowering, placement, accumulation, and shape restoration are exact integer operations. The unit adds no stochastic source of its own; modeled deviations enter only through the execution substrate.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathbf{W}$ | logical weight matrix | — | `weight` |
| $\mathbf{X}$ | logical activation matrix | — | input tensor |
| $\mathbf{Y}$ | pre-requantize integer output | — | operator return |
| $N$ | logical output width | — | weight shape |
| $K$ | contraction width | — | weight and input shape |
| $M$ | flattened operator-output positions | — | input shape |
| $b$ | optional integer bias vector | — | programmed bias |

## Assumptions, scope & validity

- Weights and activations are integers inside the published value ranges.
- Every lowering map has a matching inverse aggregation for the axes it introduces.
- Requantization is outside the unit model.

## References

TODO.
