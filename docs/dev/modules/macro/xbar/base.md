# `XbarMacro`

Abstract registry root for the XbarMacro family, declared in `neurox/macro/xbar/base.py`.

## Public surface

- `XbarMacroConfig` — frozen dataclass acting as the registry-key root. **No fields**: degenerate family members (e.g. `IdealXbarMacro`) do not own an xbar, so xbar configuration lives on the concrete subclass configs.
- `XbarMacro` — abstract `FabricateMixin + nn.Module + ProfileMixin` with:
  - `from_config(cls, *, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)` polymorphic dispatcher (uses `RegistryMixin` keyed on `type(cfg)`).
  - Base `__init__` with the same signature; records `self._w_logical_shape` and stashes the construction context (`_macro_dtype`, `_macro_T__K`, `_ideal_xbar`, `_macro_name`). The base does **not** declare an `xbar` attribute — xbar-using subclasses declare and build it themselves.
  - `self._build_xbar(*, xbar_cfg, inst_shape) -> Xbar` helper that calls `Xbar.from_config(...)` and applies `.to_ideal()` when `ideal_xbar=True`. `xbar_cfg` is passed in explicitly so the base does not presume the concrete cfg carries one.
  - Abstract `w_value_range / x_value_range / output_rescale_factor` properties.
  - Abstract `program(weight)` and `matmul(input)` (no weight in matmul — the subclass reads its own programmed state; matches `torch.matmul` semantics, bias and requantize live in the operator).
  - Static `chunk_pad_along(t, *, axis, chunk_size, pad_value)` — shared geometric helper. `pad_value` has no default per the physical-layer rule.

The base provides **no** template method for the run path. Concrete subclasses write their own `program` and `matmul` end to end; the base only owns construction scaffolding, the xbar build helper, and the static tensor helper.

## Subclasses

Xbar-using subclasses (`DirectXbarMacro`, `InterArraySliceXbarMacro`, `IntraArraySliceXbarMacro`):
- Define their own `*Config` dataclass extending `XbarMacroConfig`, declaring `xbar_cfg: XbarConfig` plus the slicer / reducer fields they need.
- Register via `@XbarMacro.register_key(MyConfig)`.
- Declare `xbar: Xbar` as an instance attribute.
- Follow the family signature `(*, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar)`; forward `super().__init__(...)`, derive the xbar's `inst_shape` symbolically from `w_logical_shape + cfg.xbar_cfg.col_num / row_num + slice counts`, then call `self._build_xbar(xbar_cfg=cfg.xbar_cfg, inst_shape=...)`. Macros never reach into xbar-subclass-specific config fields — `col_num` and `row_num` (in base `XbarConfig`) are the only cfg reads required before the xbar exists; everything else (`w_digit_count`, `w_digit_radix`, `x_range`, `w_digit_range`) is read off the constructed `self.xbar` via its abstract properties.
- Own their slicers, reducers, and the paired organize / aggregate code.
- Implement `program(weight)` to organize `weight` and call `self.xbar.program(organized)`.

The degenerate member [`IdealXbarMacro`](ideal.md) has no `xbar_cfg`, no `xbar`, no `_build_xbar` call; it just stores `weight` and does `torch.matmul`. The `ideal_xbar` arg is accepted and ignored.

See [`direct.md`](direct.md), [`inter_array_slice.md`](inter_array_slice.md), [`intra_array_slice.md`](intra_array_slice.md), and [`ideal.md`](ideal.md).
