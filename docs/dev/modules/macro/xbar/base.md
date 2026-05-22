# `XbarMacro`

Abstract base for xbar-backed macros, declared in `neurox/macro/xbar/base.py`.

## Public surface

- `XbarMacroConfig` — frozen dataclass with `xbar_cfg: XbarConfig`. Concrete subclass configs extend it with their own slicer / reducer fields and register via `_neurox_type`.
- `XbarMacro` — abstract `FabricateMixin + nn.Module + ProfiledModule` with:
  - `from_config(cls, *, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)` polymorphic dispatcher (uses `RegistryDispatchMixin` keyed on `type(cfg)`).
  - Base `__init__` with the same signature; records `self._w_logical_shape` and stashes the construction context. The base does **not** build the xbar — that is the concrete subclass's job because the xbar's `w_layout_shape` is derived from the subclass-specific organize / slice logic.
  - `self._build_xbar(xbar_w_layout_shape) -> Xbar` helper that calls `Xbar.from_config(...)` with the derived shape and applies `.to_ideal()` when `ideal_xbar=True`.
  - Abstract `w_value_range / x_value_range / output_rescale_factor` properties.
  - Abstract `program(weight)` and `matmul(input, bias, mult, rshift, zp)` (no weight in matmul — the macro reads `self.xbar`'s programmed state).
  - Static `chunk_pad_along(t, *, axis, chunk_size, pad_value)` — shared geometric helper. `pad_value` has no default per the physical-layer rule.

The base provides **no** template method for the run path. Concrete subclasses write their own `program` and `matmul` end to end; the base only owns construction scaffolding and the static helper.

## Subclasses

Each concrete xbar macro mode:
- Defines its own `*Config` dataclass extending `XbarMacroConfig` with the sub-module configs it needs.
- Registers itself via `@XbarMacro.register_key(MyConfig)`.
- Follows the family signature `(*, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)`; forwards `super().__init__(...)`, derives the xbar's `w_layout_shape` symbolically from `w_logical_shape + cfg`, then builds the xbar (via `self._build_xbar(...)`), the slicers, and the digital reducers with their derived `inst_shape`s.
- Owns its slicers, reducers, and the paired organize / aggregate code.
- Implements `program(weight)` to organize `weight` and call `self.xbar.program(organized)`.

See [`inter_xbar_slice.md`](inter_xbar_slice.md) and [`intra_xbar_slice.md`](intra_xbar_slice.md).
