# `XbarMacro`

Abstract base for xbar-backed macros, declared in `neurox/macro/xbar/base.py`.

## Public surface

- `XbarMacroConfig` — frozen dataclass with `xbar_cfg: XbarConfig`. Concrete subclass configs extend it with their own slicer / reducer fields and register via `_neurox_type`.
- `XbarMacro` — abstract `FabricateMixin + nn.Module + ProfileMixin` with:
  - `from_config(cls, *, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)` polymorphic dispatcher (uses `RegistryMixin` keyed on `type(cfg)`).
  - Base `__init__` with the same signature; records `self._w_logical_shape` and stashes the construction context. The base does **not** build the xbar — that is the concrete subclass's job because the xbar's `inst_shape` is derived from the subclass-specific organize / slice logic.
  - `self._build_xbar(inst_shape) -> Xbar` helper that calls `Xbar.from_config(..., inst_shape=inst_shape, ...)` and applies `.to_ideal()` when `ideal_xbar=True`. The xbar owns its own trailing `(col_num, w_digit_count, row_num)` dims; the macro only supplies the per-instance multiplicity prefix.
  - Abstract `w_value_range / x_value_range / output_rescale_factor` properties.
  - Abstract `program(weight)` and `matmul(input)` (no weight in matmul — the macro reads `self.xbar`'s programmed state; matches `torch.matmul` semantics, bias and requantize live in the operator).
  - Static `chunk_pad_along(t, *, axis, chunk_size, pad_value)` — shared geometric helper. `pad_value` has no default per the physical-layer rule.

The base provides **no** template method for the run path. Concrete subclasses write their own `program` and `matmul` end to end; the base only owns construction scaffolding and the static helper.

## Subclasses

Each concrete xbar macro mode:
- Defines its own `*Config` dataclass extending `XbarMacroConfig` with the sub-module configs it needs.
- Registers itself via `@XbarMacro.register_key(MyConfig)`.
- Follows the family signature `(*, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)`; forwards `super().__init__(...)`, derives the xbar's `inst_shape` symbolically from `w_logical_shape + cfg.xbar_cfg.col_num / row_num + slice counts`, then builds the xbar (via `self._build_xbar(inst_shape=...)`), the slicers, and the digital reducers with their derived `inst_shape`s. Macros never reach into xbar-subclass-specific config fields — `col_num` and `row_num` (in base `XbarConfig`) are the only cfg reads required before the xbar exists; everything else (`w_digit_count`, `w_digit_radix`, `x_range`, `w_digit_range`) is read off the constructed `self.xbar` via its abstract properties.
- Owns its slicers, reducers, and the paired organize / aggregate code.
- Implements `program(weight)` to organize `weight` and call `self.xbar.program(organized)`.

See [`direct.md`](direct.md), [`inter_array_slice.md`](inter_array_slice.md), and [`intra_array_slice.md`](intra_array_slice.md).
