# `XbarMacro`

Abstract registry root for the XbarMacro family, declared in `neurox/macro/xbar/base.py`.

## Public surface

- `XbarMacroConfig` — frozen dataclass acting as the registry-key root. **No fields**: degenerate family members (e.g. `IdealXbarMacro`) do not own an xbar, so xbar configuration lives on the concrete subclass configs.
- `XbarMacroPolicy` — empty marker base policy for the family. Concrete macro impls declare their own `*Policy(XbarMacroPolicy)` (e.g. `DirectXbarMacroPolicy` with an `xbar: XbarPolicy` field, or empty `IdealXbarMacroPolicy()` for the degenerate member). The composite that holds an XbarMacro stores the abstract `XbarMacroPolicy` field type and the caller passes the concrete impl.
- `XbarMacro` — abstract `FabricateMixin + nn.Module + ProfileMixin` with:
  - `from_config(cls, *, config, policy, name, w_logical_shape, dtype, T__K, ideal_xbar)` polymorphic dispatcher (uses `RegistryMixin` keyed on `type(config)`).
  - Base `__init__` with the same signature; records `self._w_logical_shape` and stashes the construction context (`_macro_dtype`, `_macro_T__K`, `_ideal_xbar`, `_macro_name`). The base does **not** declare an `xbar` attribute — xbar-using subclasses declare and build it themselves.
  - `self._build_xbar(*, xbar_config, xbar_policy, inst_shape) -> Xbar` helper that calls `Xbar.from_config(...)` and applies `.to_ideal()` when `ideal_xbar=True`. When `ideal_xbar=True` the passed `xbar_policy` is discarded in favour of `IdealXbarPolicy()`. `xbar_config` and `xbar_policy` are passed in explicitly so the base does not presume the concrete config carries them.
  - Abstract `w_value_range / x_value_range` value-grid properties and `adc_mode_num / adc_max_bits` ADC-surface properties; abstract method `adc_rescale_factor(adc_operation_point) -> float`.
  - Abstract `program(weight)` and `matmul(input, *, adc_operation_point)` (no weight in matmul — the subclass reads its own programmed state; matches `torch.matmul` semantics, bias and requantize live in the operator).
  - Static `chunk_pad_along(t, *, axis, chunk_size, pad_value)` — shared geometric helper. `pad_value` has no default per the physical-layer rule.

The base provides **no** template method for the run path. Concrete subclasses write their own `program` and `matmul` end to end; the base only owns construction scaffolding, the xbar build helper, and the static tensor helper.

## Subclasses

Every xbar-using subclass:
- Define their own `*Config` dataclass extending `XbarMacroConfig`, declaring `xbar_config: XbarConfig` plus the slicer / reducer fields they need.
- Define their own `*Policy(XbarMacroPolicy)` carrying an `xbar: XbarPolicy` field (and any future macro-local switches).
- Register via `@XbarMacro.register_key(MyConfig)`.
- Declare `xbar: Xbar` as an instance attribute.
- Follow the family signature `(*, config, policy, name, w_logical_shape, dtype, T__K, ideal_xbar)`; forward `super().__init__(...)`, derive the xbar's `inst_shape` symbolically from `w_logical_shape + config.xbar_config.col_num / row_num + slice counts`, then call `self._build_xbar(xbar_config=config.xbar_config, xbar_policy=policy.xbar, inst_shape=...)`. Macros never reach into xbar-subclass-specific config fields — `col_num` and `row_num` (in base `XbarConfig`) are the only config reads required before the xbar exists; everything else (`w_digit_count`, `w_digit_radix`, `x_range`, `w_digit_range`) is read off the constructed `self.xbar` via its abstract properties.
- Own their slicers, reducers, and the paired organize / aggregate code.
- Implement `program(weight)` to organize `weight` and call `self.xbar.program(organized)`.

The degenerate member [`IdealXbarMacro`](ideal.md) has no `xbar_config`, no `xbar`, no `_build_xbar` call, and an empty `IdealXbarMacroPolicy()`; it just stores `weight` and does `torch.matmul`. The `ideal_xbar` arg is accepted and ignored.

See [`direct.md`](direct.md), [`inter_array_slice.md`](inter_array_slice.md), [`intra_array_slice.md`](intra_array_slice.md), and [`ideal.md`](ideal.md).
