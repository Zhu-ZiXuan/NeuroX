# Quantization Kernels — Implementation

## Summary

`neurox/common/quant.py` holds the small set of stochastic-rounding quantization primitives shared across the codebase. Three kernels make up the surface: `stochastic_floor_div` (floor right-shift integer division), `stochastic_floor_to_int` (float-to-int floor quantizer $\operatorname{code} = \lfloor \operatorname{signal} \cdot \operatorname{scale} \rfloor$), and `floor_bucketize` (bucketize against code-edge boundaries with floor semantics). All three are stateless math: none of them know about any consumer. Each consumes a `training: bool` directly; when set, an unbiased one-LSB uniform jitter is added before the floor, so the same code path serves deterministic conversion (`training=False`) and stochastic-rounded conversion (`training=True`) without moving where the boundaries sit. There is no physical Reference spec — these are quantization math, not a device model.

## Design decisions

- **`training` is a direct argument, not an internal flag.** Each kernel takes `training: bool` rather than reading state. The flag is sourced from the calling module's training mode: standard PyTorch training mode is the single source of truth for stochastic-versus-deterministic, and there is deliberately no per-call override knob. This keeps the kernels stateless and lets the caller's mode drive dithering uniformly.
- **Floor placement, not round-to-nearest.** `floor_bucketize` puts boundaries at code edges $B_c = c \cdot \operatorname{LSB}$, so a signal in $[B_c,\ B_{c+1})$ quantizes to code $c$. Floor semantics mean the boundaries do not move between deterministic and stochastic mode — only the pre-floor jitter changes. Round-to-nearest would place boundaries at bin centres and require a different deterministic and stochastic split.
- **Jitter is added before the floor, sized to one LSB.** Adding an unbiased $\operatorname{uniform}(0, \operatorname{LSB})$ offset (one full bin width) pre-floor makes the expected output code equal the un-quantized value: stochastic rounding is unbiased. Each kernel parameterizes the LSB in its own units — `floor_bucketize` takes `lsb` explicitly, `stochastic_floor_to_int` works in code space where one LSB is unit width (jitter is $\operatorname{uniform}(0,1)$ in scaled units), and `stochastic_floor_div` works in integer numerator space where one LSB is $2^{\operatorname{rshift}}$.
- **`stochastic_floor_div` branches on scalar-versus-tensor `rshift`.** A scalar shift draws integer jitter directly in the numerator's integer dtype over $[0,\ 2^{\operatorname{rshift}})$. A tensor-valued (per-element) shift instead samples a float jitter in $[0,1)$, scales by the per-element denominator $2^{\operatorname{rshift}}$ in wide float, and casts back to the integer dtype — keeping the result exact-modulo-denominator while supporting broadcast shifts.

## Contracts & invariants

- **Output dtype.** `stochastic_floor_div` returns the same dtype as `numerator`. `stochastic_floor_to_int` and `floor_bucketize` return `out_dtype` (an integer dtype passed by the caller).
- **`boundaries` are sorted ascending of length $n_{\operatorname{codes}} - 1$.** `floor_bucketize` returns a code in $[0,\ n_{\operatorname{codes}} - 1]$. It bucketizes with right-of-boundary placement: a signal exactly on a boundary ($\operatorname{signal} = c \cdot \operatorname{LSB}$) lands in the upper bin and yields code $c$, which is what produces floor semantics. The opposite placement would round-to-nearest at boundaries.
- **`scale` is codes per signal unit.** For `stochastic_floor_to_int`, `scale` is the reciprocal of one LSB step in `signal`'s units, i.e. multiplying maps the physical signal into code space before the floor.
- **Eval mode is bit-exact and deterministic.** With `training=False` every kernel reduces to a plain floor / right-shift / bucketize with no random draw, so repeated calls on identical input return identical output.
- **Train mode is statistically unbiased.** With `training=True` the sample mean of the output converges to the true (un-floored) quotient or code. This is the defining property the kernels guarantee, exercised by `tests/test_stochastic_rounding.py`.
- **Stochastic-rounding jitter can overshoot the legal range.** Because jitter is added pre-floor, a value near the top boundary can produce a code one above the nominal maximum. Callers that need a bounded code must clamp the result themselves; the kernels do not clamp.

## Performance & resources

- **Stateless and allocation-light.** No persistent buffers; the only allocations are the per-call jitter tensors, drawn to match the input shape and device so they stay on-device. The wide-float denominator in the tensor-`rshift` branch of `stochastic_floor_div` is the one wide-dtype temporary.
- **Per-element cost is $O(1)$ over the input; no reductions across elements.** The whole surface is elementwise (modulo the boundary search in `floor_bucketize`, which is $O(\log n_{\operatorname{codes}})$ per element).
- **`rshift` branch is trace-time, not value-dependent.** When compiled, the scalar-versus-tensor `rshift` branch keys on the Python type of the argument (resolved at trace time), not on a tensor value, so it stays [dynamo-safe](../compile/contracts.md); the random draws are traceable.

## Gotchas

- **The LSB units differ per kernel.** `floor_bucketize` jitter is in the signal's physical units (sized by `lsb`); `stochastic_floor_to_int` jitter is in code space (unit LSB, so `uniform(0,1)`); `stochastic_floor_div` jitter is in integer numerator space (LSB $= 2^{\operatorname{rshift}}$). Passing a wrong LSB silently biases the rounding rather than erroring.
- **Right-of-boundary placement is load-bearing.** Flipping `floor_bucketize` to the opposite placement silently switches from floor to round-to-nearest at boundaries and breaks alignment with the code-edge convention $B_c = c \cdot \operatorname{LSB}$.
- **No clamp at the kernel.** Forgetting the caller-side clamp after `training=True` lets a top-of-range value emit an out-of-range code.

## Known limitations

- **No symmetric / signed rounding mode.** The kernels are floor-only; a sign-aware (round-toward-zero) variant is not provided. Signed-code consumers handle the sign via a downstream zero-code shift, not here.
- **The tensor-`rshift` path widens to wide float.** Per-element shifts pay a wide-float temporary for the denominator; a fully-integer per-element jitter would avoid it but is not implemented.

---

- **Reference**: N/A — quantization math, no physical spec.
- **Implementation**: `neurox/common/quant.py`
- **Tests**: `tests/test_stochastic_rounding.py`
- **Decisions**: N/A.
