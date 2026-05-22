# `neurox/xbar/base.py`

## Current role

`Xbar` is the abstract base for the physical-crossbar primitive. Every concrete xbar (offset-coded 1T1R, future differential 1T1R, ideal twin) inherits this class and implements the same primitive shape contract. The family uses `RegistryMixin[type[XbarConfig], Xbar]`; concrete subclasses register on their concrete config type via `@Xbar.register_key(<Subclass>Config)`.

`Xbar.from_config(cls, *, cfg, name, w_layout_shape, dtype, T__K)` is the family constructor: it looks up the impl class from `type(cfg)` and forwards the runtime arguments. Direct instantiation of a concrete subclass is allowed; `from_config` is the polymorphic entry that owning macros use.

`XbarConfig` is the base config carrying tile geometry, the runtime ADC operating point, the `(adc_mode, adc_bits) → rescale_factor` lookup, and tile-level PPA fields. All fields are required; the physical-layer no-defaults rule applies.

## Primitive shape contract

The generic xbar publishes exactly two shape contracts:

- `program(w)` — weight digit tensor whose shape matches the `w_layout_shape` bound at `__init__`: `(*prefix, data_num, digit_num, row_num)`. Every entry must lie in `w_digit_range`.
- `vec_mat_mul(x)` — activation tensor with primitive trailing dims `[row_num]`; returns an output tensor with primitive trailing dims `[data_num]`. Entries of `x` must lie in `x_range`.

These are the **only** shape semantics the generic xbar exposes. Any additional leading axes wrapped around `(*data_num, digit_num, row_num)` are broadcast against the fabricated per-cell state without further interpretation by the xbar.

## Topology-agnostic row / column

The base defines row and column axes by **function**, not by any specific wiring scheme:

- a **row** is the set of cells that share the same input `x`;
- a **column** is the set of cells whose contributions aggregate into one output.

For families where these happen to land on physical WL / BL lines the mapping lives inside that family's own modules — not in this generic contract.

## Value-domain capabilities

Every concrete xbar exposes:

- `x_range` — single-cycle integer input grid the tile can carry (encoding-independent).
- `w_digit_count` — number of digits per xbar-word.
- `w_digit_radix` — positional base `r` of the in-tile digit combination.
- `w_digit_range` — inclusive integer range a single digit cell can carry physically. Set by the array structure and the device's state count.

The xbar does **not** expose an aggregate "full logical `w` range" — that range depends on the transcoder / slicing / signed-digit policy applied above the xbar layer.

## Output rescale lookup

`XbarConfig.output_rescale_factors` is an externally-calibrated table of `XbarRescaleEntry(adc_mode, adc_bits, rf)` rows. `Xbar.output_rescale_factor` returns the entry matching the runtime `(adc_mode, adc_bits)` operating point. See [`docs/dev/architecture/mapping.md`](docs/dev/architecture/mapping.md) for the surrounding flow.

## `to_ideal()`

`Xbar.to_ideal() -> IdealXbar` is a **concrete** base method. It auto-forwards every `XbarConfig` field from `self.cfg` plus the four abstract structural properties (`x_range`, `w_digit_*`) into an `IdealXbarConfig`, derives the new tile's `w_layout_shape = (*self._inst_shape, col_num, w_digit_count, row_num)`, then instantiates `IdealXbar`. Concrete xbars do not override it; `IdealXbar.to_ideal()` overrides to `return self`. `IdealXbar` is registered on `IdealXbarConfig`, so the same lossless tile is reachable through either `physical.to_ideal()` or `Xbar.from_config(cfg=IdealXbarConfig(...))`.

## Lifecycle

`Xbar` inherits `FabricateMixin`. The split is:

- `__init__` stores `self.cfg`, `self.T__K`, `self.dtype`, and validates / records `self._w_layout_shape`. Sub-modules (when present, e.g. `core` + `readout` in `Offset1T1RXbar`) are constructed here with derived shapes.
- `fabricate()` is the inherited auto-cascade. Each concrete xbar overrides `_sample_fabricate_mismatch` only if it owns mismatch state directly; in the offset-1T1R lineage the actual state lives in the device children, so `_sample_fabricate_mismatch` is the default no-op and the cascade fans into the children.
- `program(w)` (abstract) writes the programmed digit state. `IdealXbar` reassigns `self.digits = w`; physical xbars push `w` through their slicer / reference-column scatter and call the underlying core's `program(...)`.
- `vec_mat_mul(x)` is the pure-forward read.

Stochastic-vs-deterministic rounding inside the quantisers downstream of the xbar tracks `self.training`; there is no separate `stochastic` knob.

See also:

- `ideal.md`
- `docs/dev/modules/xbar/_1t1r/README.md`
