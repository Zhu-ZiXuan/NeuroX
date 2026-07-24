# Quantization primitives

`neurox/common/quant.py` is the shared quantization toolbox: stochastic-rounding integer conversion, fixed-point scale decomposition, min/max observers, and straight-through fake-quantize. The observers hold EMA buffers; the remaining operations are stateless.

## Design decisions

- **`training` is the single dithering switch.** Each stochastic kernel takes `training: bool` sourced from the calling module's `self.training`; standard PyTorch train/eval mode is the only source of truth for stochastic-versus-deterministic conversion — there is no per-call override and no separate nonideality policy switch.
- **Floor placement, not round-to-nearest.** `floor_bucketize` uses floor semantics so the code boundaries stay fixed between deterministic and stochastic mode — only the pre-floor jitter moves. Round-to-nearest would place boundaries at bin centers and demand a different split between the two modes.
- **Jitter is pre-floor and one LSB wide, for unbiasedness.** An unbiased $\operatorname{uniform}(0, \mathrm{LSB})$ offset added before the floor makes the expected output code equal the un-quantized value. Each kernel parameterizes the LSB in its own units, so the dither stays one bin wide wherever the kernel operates.
- **Two observers, two grids.** The per-tensor observer tracks an asymmetric affine range with a computed `zero_point`, because activation distributions are skewed; the per-channel observer tracks a symmetric range per output channel and pins `zero_point = 0`. The symmetric grid keeps the integer matmul free of a zero-point cross-term, and per-channel granularity captures each filter's own dynamic range.
- **Observer state lifecycle.** Each observer is a stateful `nn.Module` whose `forward` is a watcher — it mutates buffers and returns nothing rather than transforming its input, while a separate `qparams()` reads the state out. An EMA of the min/max (or abs-max) accumulates while the module is training and un-frozen; `freeze()` pins it for the inference phase, and because `frozen` is a registered buffer the pinned stats ride the `state_dict` and outlive later `train()` / `eval()` toggles. The initial `inf` buffer is an uninitialized sentinel: the first observed batch is copied in rather than blended, so no infinity pollutes the EMA.
- **Two rounding modes for two roles.** The stochastic kernels floor with unbiased dither because the forward must emit a real integer code while training still sees the right mean; the fake-quantize helpers instead round-to-nearest and let the gradient pass straight through the round (the straight-through estimator), the mode a QAT step wants. The two rounding rules are deliberately different.
- **Fixed-point conversion and the shift kernel compose.** `derive_multiplier_and_shift_tensor` emits the `rshift` that `stochastic_floor_div` consumes, so the two families are one integer-rescale pipeline rather than independent helpers. The multiplier precision is bounded to keep the rescaled product inside the integer accumulator budget — the `DEFAULT_MULT_BITS` note in code owns the exact bound.

## Contracts & invariants

- **Eval is bit-exact; train is unbiased.** With `training=False` every stochastic kernel reduces to a plain floor / shift / bucketize with no random draw, so repeated calls on identical input return identical output. With `training=True` the sample mean of the output converges to the un-floored quotient or code.
- **Kernels never clamp.** Because jitter is added before the floor, a value near the top boundary can emit a code one above the nominal maximum. Output bounds are not part of these kernel contracts.
- **The scalar-versus-tensor `rshift` branch resolves at trace time.** `stochastic_floor_div` keys the branch on the Python type of `rshift`, fixed at trace time rather than on a tensor value, so it stays [dynamo-safe](../compile/contracts.md); the random draws are traceable.

## Gotchas

- **Default `training=True` dithers every call.** Because dithering follows the owning module's `self.training` alone, a freshly constructed module (default `training=True`) yields a non-deterministic conversion per call — even with every nonideality policy off. Call `.eval()` on the owning module for the deterministic, bit-exact path.
- **The LSB units differ per kernel.** `floor_bucketize` jitter is in the signal's physical units (sized by `lsb`), `stochastic_floor_to_int` jitter is in unit-LSB code space, and `stochastic_floor_div` jitter is in integer space where one LSB is $2^{\mathrm{rshift}}$. A wrong LSB silently biases the rounding rather than erroring.

---

- **Reference**: N/A — software utility.
- **Implementation**: `neurox/common/quant.py`
- **Tests**: `tests/common/test_stochastic_rounding.py` (stochastic-rounding kernels); TODO — no dedicated observer, fake-quant, or fixed-point tests.
