# Unit family

## Shared conventions

A unit realizes one integer tensor operator over its accepted weight and activation ranges, returning the result before requantization. Requantization to an activation grid lies outside the model.

Mapping restores every representation axis it introduces. Linear input-leading axes remain unchanged; convolution windows return to the output spatial layout. An integer bias is added after substrate aggregation and introduces no execution cycle or dynamic-energy event.

## Governing laws

For a logical weight matrix $\mathbf{W}$ of shape $(N, K)$ and an activation matrix $\mathbf{X}$ of shape $(M, K)$,

$$\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}, \qquad Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

with both operands restricted to the unit's accepted integer value ranges. The operator-specific maps arrange the operands around this contraction and restore the declared output shape.

If the contraction dimension is partitioned, results from its partitions are accumulated. Output blocks are not reduced; they are restored to logical output order.

For internal operations with durations $t_i$, temporal reuse completes in $t_{\mathrm{serial}}=\sum_i t_i$. A stage of simultaneously started physical replicas completes in $t_{\mathrm{parallel}}=\max_i t_i$. A sequence of stages with completion barriers sums those stage durations. A unit latency covers one basic operation: one VMM for a linear unit, or one complete image for a convolution unit. The operation includes its internal pipelines. Batch, token and timestep counts belong to the caller, which combines actual operations independently for each sample.

## Noise & non-idealities

Lowering and shape restoration are exact integer operations. The unit adds no stochastic source; analog, conversion, and register-width effects belong to the execution substrate.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $\mathbf{W}$ | logical weight matrix | — | `weight` |
| $\mathbf{X}$ | logical activation matrix | — | input tensor |
| $\mathbf{Y}$ | pre-requantize integer output | — | operator return |
| $N$ | logical output width | — | weight shape |
| $K$ | contraction width | — | weight and input shape |
| $M$ | flattened operator-output positions | — | input shape |
| $m,n,k$ | input-row, output, and contraction indices | — | — |
| $t_i$ | duration of one internal operation | ns | — |
| $t_{\mathrm{serial}}$ | duration of a serial sequence | ns | — |
| $t_{\mathrm{parallel}}$ | duration of a parallel stage | ns | — |

## References

TODO.
