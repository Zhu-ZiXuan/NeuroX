# `neurox/common/quant.py`

## Current role

`quant.py` holds the small set of quantisation primitives shared across the NeuroX codebase.

## Public surface

- `floor_bucketize(value, boundaries, *, out_dtype, training, lsb)` — floor-style bucketize. Boundaries placed at code edges `B_c = c · LSB`; a signal in `[B_c, B_{c+1})` quantises to code `c`.
- `stochastic_floor_div(numerator, rshift, *, training)` — floor right-shift for integer division.
- `stochastic_floor_to_int(signal, scale, *, out_dtype, training)` — floor quantisation from float to int: `code = floor(signal · scale)` where `scale` is codes per `signal` unit (reciprocal of one LSB step).

Every helper consumes `training: bool` directly. The flag is sourced from the calling `nn.Module`'s `self.training`: when `True`, an unbiased one-LSB uniform jitter is added before the floor; when `False`, the helper reduces to a plain floor / right-shift. There is no override knob — the single source of truth for stochastic-vs-deterministic is the standard PyTorch training mode.

## Floor semantics

The bucketize kernel uses floor placement so the same code handles deterministic conversion (`training=False`) and stochastic-rounded conversion (`training=True`) without changing where the boundaries sit. The LSB jitter is added pre-floor so the expected code value remains unbiased.
