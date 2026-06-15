# `neurox/xbar/base.py`

## Current role

`Xbar` is the abstract base for the physical-crossbar primitive. Every concrete xbar (offset-coded 1T1R, future differential 1T1R, ideal twin) inherits this class and implements the same primitive shape contract. The family uses `RegistryMixin[type[XbarConfig], Xbar]`; concrete subclasses register on their concrete config type via `@Xbar.register_key(<Subclass>Config)`.

`Xbar.from_config(cls, *, config, policy, name, inst_shape, dtype, T__K)` is the family constructor: it looks up the impl class from `type(config)` and forwards the runtime arguments. Direct instantiation of a concrete subclass is allowed; `from_config` is the polymorphic entry that owning macros use.

`XbarConfig` is the base config carrying tile geometry and the xbar's **own extra PPA** (wiring + control glue at the xbar boundary — does **not** include `core` or `readout`, which carry their own PPA and roll up independently via the composite-aggregation rule in [`profiler_and_ppa.md`](../../dev/architecture/profiler_and_ppa.md)). It does **not** carry the chip-calibrated `(adc_mode, adc_bits) → rescale_factor` table; that field lives on the concrete physical config (`Offset1T1RXbarConfig.adc_calibration`). The ideal twin derives its rescale from `adc_bits` and integer geometry alone, with no calibration input. The active `AdcOperationPoint` is threaded into `vec_mat_mul`; every xbar exposes `adc_mode_num` / `adc_max_bits` (delegated from its ADC / readout chain or carried directly on the ideal config) and `adc_rescale_factor(adc_operation_point) -> float`. All fields are required; the physical-layer no-defaults rule applies.

`XbarPolicy` is the empty marker base policy for the family. Concrete xbar impls declare their own structured `*Policy(XbarPolicy)` (e.g. `Offset1T1RXbarPolicy`, `IdealXbarPolicy`) carrying nested sub-policies for every child the impl owns. The composite that holds an Xbar stores the abstract `XbarPolicy` field type and the caller passes the concrete impl.

## Primitive shape contract

The generic xbar publishes exactly two shape contracts:

- `program(w)` — weight digit tensor whose shape matches `self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)`. The trailing three dims are owned by the xbar (derived from config and the subclass-specific structural properties); only `inst_shape` is supplied at construction. Every entry must lie in `w_digit_range`.
- `vec_mat_mul(x, *, adc_operation_point)` — activation tensor with primitive trailing dims `[row_num]`; returns an output tensor with primitive trailing dims `[data_num]`. Entries of `x` must lie in `x_range`; `adc_operation_point` selects the ADC operating point used for output digitisation.

These are the **only** shape semantics the generic xbar exposes. Leading dims on each method serve different roles:

- `program(w)` — additional leading dims around the trailing `(col_num, w_digit_count, row_num)` broadcast into `inst_shape` positions (one fab instance per inst slot).
- `vec_mat_mul(x)` — additional leading dims around the trailing `[row_num]` express **external batch and inst alignment only**. Per-column / per-digit / per-phys_col fanout is an **internal xbar structural detail** — the implementation opens those axes itself (via `unsqueeze(-2)` against the fabricated `g` grid). Callers must not encode column / digit / phys_col positions in `x`'s leading dims; doing so would collide with the xbar's own internal axes during broadcast and is a contract violation.

### Leading-axis alignment for weight-instance tiles

When an xbar has non-empty `inst_shape` (multiple fabricated tile instances on chip), the caller must arrange `x`'s leading dims so they broadcast cleanly against the inst structure. Two intended patterns:

- **Caller wants one x shared across all inst** (e.g. evaluation sweeps): pass `x` with a **size-1 broadcast slot** at the inst position(s). For a single-axis `inst_shape=(K,)` and `x_batch` independent inputs, that means `x.shape = (x_batch, 1, row_num)` — the `1` lets the K inst copies broadcast into the leading slot at runtime.
- **Caller wants one x per inst** (e.g. macro paths that have one input per fab instance): supply `x.shape = (*x_batch, *inst_shape, row_num)` directly with matching inst dims.

This convention is the xbar's interface contract — not an internal detail. Tools that drive the xbar from raw `(x_batch, row_num)` tensors (e.g. ADC calibration, statistic) typically `x.unsqueeze(-2)` to obtain the broadcast slot; macros (Direct / IntraArraySlice / InterArraySlice) construct `x` with the right leading layout via their `_organize_x` helper.

**Warning — silent semantic flip when shapes happen to match.** If the caller omits the size-1 inst slot AND `x_batch` happens to equal an inst dim's size `K`, PyTorch's broadcast does **not** raise: it interprets that position as a *matched* axis ("one x per inst", one-to-one) instead of the intended Cartesian "x_batch × inst" broadcast. The two semantics produce different output shapes and physically different VMM workloads, but no error is reported and the silent flip is easy to miss. Always insert an explicit `1` slot at the inst position when you want Cartesian broadcast, regardless of whether `x_batch == K` happens to coincide.

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

`Xbar.adc_rescale_factor(adc_operation_point)` is `@abstractmethod` on the base; each concrete xbar owns its mapping. Physical impls (e.g. `Offset1T1RXbar`) carry an externally-calibrated `adc_calibration` table of `AdcCalibrationRecord(adc_mode, adc_bits, rescale_factor)` rows on their concrete config and look the rescale up keyed by the runtime `(adc_mode, adc_bits)` pair. `IdealXbar` derives the rescale at construction time from `adc_bits` and integer geometry alone — no chip data enters — and ignores `adc_mode`. See [`docs/dev/architecture/mapping.md`](../../dev/architecture/mapping.md) for the surrounding flow.

## `to_ideal()`

`Xbar.to_ideal() -> IdealXbar` is a **concrete** base method. It auto-forwards every `XbarConfig` field from `self.config` (geometry + PPA — `adc_calibration` is **not** an `XbarConfig` field and is intentionally not carried over) plus the structural properties (`x_range`, `w_digit_*`) and the ADC surface (`adc_mode_num`, `adc_max_bits`) into an `IdealXbarConfig`, constructs an empty `IdealXbarPolicy()`, then instantiates `IdealXbar` with the source tile's `inst_shape`. Concrete xbars do not override it; `IdealXbar.to_ideal()` overrides to `return self`. `IdealXbar` is registered on `IdealXbarConfig`, so the same lossless tile is reachable through either `physical.to_ideal()` or `Xbar.from_config(config=IdealXbarConfig(...), policy=IdealXbarPolicy(), ...)`.

## Lifecycle

`Xbar` inherits `FabricateMixin`. The split is:

- `__init__` stores `self.config`, `self.T__K`, `self.dtype`, and records `self._inst_shape`. `self._w_layout_shape` is exposed as a property derived from `inst_shape + config + subclass geometry`. Sub-modules (when present, e.g. `core` + `readout` in `Offset1T1RXbar`) are constructed here with derived shapes.
- `fabricate()` is the inherited auto-cascade. Each concrete xbar overrides `_sample_fabricate_mismatch` only if it owns mismatch state directly; in the offset-1T1R lineage the actual state lives in the device children, so `_sample_fabricate_mismatch` is the default no-op and the cascade fans into the children.
- `program(w)` (abstract) writes the programmed digit state. `IdealXbar` reassigns `self.digits = w`; physical xbars push `w` through their slicer / reference-column scatter and call the underlying core's `program(...)`.
- `vec_mat_mul(x, *, adc_operation_point)` is the pure-forward read.

Stochastic-vs-deterministic rounding inside the quantisers downstream of the xbar tracks `self.training`; there is no separate `stochastic` knob.

See also:

- `ideal.md`
- `docs/modules/xbar/_1t1r/README.md`
