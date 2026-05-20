# `XbarMacro`

Abstract base for xbar-backed macros, declared in `neurox/macro/xbar/base.py`.

## Public surface

- `XbarMacroConfig` — empty frozen dataclass; concrete subclass configs extend it with their own fields and register via `_neurox_type`.
- `XbarMacro` — abstract `nn.Module + ProfiledModule` with:
  - `from_config(cls, *, cfg, xbar, name="")` polymorphic dispatcher (uses `RegistryDispatchMixin` keyed on `type(cfg)`).
  - Abstract `w_value_range / x_value_range / output_rescale_factor` properties.
  - Abstract `fabricate(weight)` and `matmul(input, weight, bias, mult, rshift, zp)`.
  - Static `chunk_pad_along(t, *, axis, chunk_size, pad_value=0.0)` — shared geometric helper.

The base provides **no** template method.  Concrete subclasses write their own `fabricate` and `matmul` end to end.

## Subclasses

Each concrete xbar macro mode:
- Defines its own `*Config` dataclass extending `XbarMacroConfig` with the sub-module configs it needs.
- Registers itself via `@XbarMacro.register_key(MyConfig)`.
- Owns its slicers, reducers, and the paired organize / aggregate code.

See [`inter_xbar_slice.md`](inter_xbar_slice.md) and [`intra_xbar_slice.md`](intra_xbar_slice.md).
