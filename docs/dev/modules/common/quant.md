# `neurox/common/quant.py`

## Current role

`quant.py` holds the small set of quantisation primitives shared across the NeuroX codebase.

## Public surface

- `floor_bucketize(value, boundaries)` — floor-style bucketize. Boundaries placed at code edges `B_c = c · LSB`; a signal in `[B_c, B_{c+1})` quantises to code `c`.
- `stochastic_floor_div(numerator, denominator)` — stochastic rounding for integer division.
- `stochastic_floor_to_int(value)` — stochastic rounding from float to int.
- `use_stochastic(stochastic_flag, training)` — resolution helper that combines an explicit `stochastic` kwarg with a `training` flag and returns the boolean any rounding kernel should branch on.

## Floor semantics

The bucketize kernel uses floor placement so the same code handles deterministic conversion (`stochastic=False`) and stochastic-rounded conversion (`stochastic=True`) without changing where the boundaries sit. The LSB jitter is added pre-floor so the expected code value remains unbiased.
