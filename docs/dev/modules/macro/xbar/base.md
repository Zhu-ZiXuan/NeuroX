# `XbarMacro`

Abstract base for xbar-backed macros, declared in `neurox/macro/xbar/base.py`.

## Public surface

- `XbarMacroConfig` — frozen dataclass with `xbar_cfg: XbarConfig`. Concrete subclass configs extend it with their own slicer / reducer fields and register via `_neurox_type`.
- `XbarMacro` — abstract `nn.Module + ProfiledModule` with:
  - `from_config(cls, *, cfg, name, T__K, dtype, ideal_xbar)` polymorphic dispatcher (uses `RegistryDispatchMixin` keyed on `type(cfg)`).
  - Base `__init__` with the same signature; builds `self.xbar = Xbar.from_config(cfg.xbar_cfg, ...)` and, if `ideal_xbar=True`, replaces it with `self.xbar.to_ideal()`.
  - Abstract `w_value_range / x_value_range / output_rescale_factor` properties.
  - Abstract `fabricate(weight)` and `matmul(input, weight, bias, mult, rshift, zp)`.
  - Static `chunk_pad_along(t, *, axis, chunk_size, pad_value)` — shared geometric helper. `pad_value` has no default per the physical-layer rule.

The base provides **no** template method for the run path. Concrete subclasses write their own `fabricate` and `matmul` end to end; the base only owns construction (xbar build + ideal swap) and the static helper.

## Subclasses

Each concrete xbar macro mode:
- Defines its own `*Config` dataclass extending `XbarMacroConfig` with the sub-module configs it needs.
- Registers itself via `@XbarMacro.register_key(MyConfig)`.
- Follows the family signature `(*, cfg, name, T__K, dtype, ideal_xbar)`; forwards `super().__init__(...)` and then builds its slicers and reducers from the cfg.
- Owns its slicers, reducers, and the paired organize / aggregate code.

See [`inter_xbar_slice.md`](inter_xbar_slice.md) and [`intra_xbar_slice.md`](intra_xbar_slice.md).
