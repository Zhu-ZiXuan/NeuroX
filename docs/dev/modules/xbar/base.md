# `neurox/xbar/base.py`

## Current role

`Xbar` is the abstract base for the physical-crossbar primitive. Every concrete xbar (offset-coded 1T1R, future differential 1T1R, ideal twin) inherits this class and implements the same primitive shape contract. The family uses `RegistryDispatchMixin[type[XbarConfig], Xbar]`; concrete subclasses register on their concrete config type via `@Xbar.register_key(<Subclass>Config)`.

`Xbar.from_config(cls, *, cfg, name, T__K, dtype)` is the family constructor: it looks up the impl class from `type(cfg)` and forwards the runtime trio. Direct instantiation of a concrete subclass is allowed; `from_config` is the polymorphic entry that owning macros use.

`XbarConfig` is the base config carrying tile geometry, the runtime ADC operating point, the `(adc_mode, adc_bits) → rescale_factor` lookup, and tile-level PPA fields. All fields are required; the physical-layer no-defaults rule applies.

## Primitive shape contract

The generic xbar publishes exactly two shape contracts:

- `fabricate(w)` — weight digit tensor with primitive trailing dims `[data_num, digit_num, row_num]`. Every entry must lie in `w_digit_range`.
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

`Xbar.to_ideal() -> IdealXbar` is a **concrete** base method. It auto-forwards every `XbarConfig` field from `self.cfg` plus the four abstract structural properties (`x_range`, `w_digit_*`) into an `IdealXbarConfig`, then instantiates `IdealXbar`. Concrete xbars do not override it; `IdealXbar.to_ideal()` overrides to `return self`. `IdealXbar` is registered on `IdealXbarConfig`, so the same lossless tile is reachable through either `physical.to_ideal()` or `Xbar.from_config(cfg=IdealXbarConfig(...))`.

The base also owns the runtime trio. `Xbar.__init__` stores `self.cfg`, `self.T__K`, and `self.dtype`; subclasses receive these through the family signature and pass them through `super().__init__(...)` without re-assigning. Subclass-specific state (digit-weight tables, owned children, lookup buffers) is the only thing a subclass init has to write. Stochastic-vs-deterministic rounding inside the quantisers downstream of the xbar tracks `self.training`; there is no separate `stochastic` knob.

See also:

- `ideal.md`
- `docs/dev/modules/xbar/_1t1r/README.md`
