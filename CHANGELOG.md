# Changelog

## Unreleased — Core as Pure Array + Parallel-Rail Solver + Scheme Extraction

### Changed

- **The crossbar core is demoted to a pure array; drivers, reference, and readout are hoisted to the scheme xbar.** `CircuitCore1T1R` becomes `Core1T1R` (`neurox/xbar/core/_1t1r.py`) and holds only the cell array, the wire parasitics, and the solver — it no longer owns the WL DAC, BL clamp, SL driver, boundary reference, or readout. Its `cim_read(x) -> Tensor` becomes `solve_array(v_wl, *, bl_driver, bl_v_ref__V, sl_driver, sl_v_ref__V) -> CoreSteadyState(i_bl_port__uA, v_bl_clamp__V)`: the input is the analog word-line drive (the scheme xbar runs the DAC), the boundary clamp drivers and their scalar reference taps are passed in, and the per-line BL port current and clamp voltage are returned for a downstream readout. The chunked serialization and the array-energy model stay in the core; the hardcoded `isinstance(OpAmpTIA)` boundary-driver assertion is gone (the core is driver-agnostic). The scheme xbar now owns `wl_dac` / `bl_clamp` / `sl_driver` / `clamp_ref` / the readout leaves / the `core` as flat peers and orchestrates them in `vec_mat_mul`.
- **The DC solver is the parallel-BL/SL solver, with the rail orientation a construction-time axis.** `NestedSolver` / `NestedSolverConfig` become `NestedParallelRailSolver` / `NestedParallelRailSolverConfig`. `Solver.from_config` gains a `series_axis: int = -1` (supplied by the core, not a config / TOML field) naming which of the two trailing cell-grid axes is the series (wire-ladder) direction; the solver canonicalises series-last at entry and restores the caller's layout at exit, with `series_axis == -1` a byte-identical no-op and a compile-time constant under `@torch.compile`. The four structural assumptions the formulation rests on (two rails, a single signed two-terminal cell branch, a driven control line, one shared series axis) are documented; the orthogonal BL ⊥ SL (2-D mesh) case is explicitly a future, separate solver.

### Removed

- **`ideal_1t1r` is deleted.** `neurox/xbar/works/ideal_1t1r/` (the hand-built ideal-driver `IdealCore1T1R` + `IdealOffset1T1RXbar`) and its configs, tests, and docs are removed; it was a provisional structure pending a proper re-model. The `IdealXbar` arithmetic oracle (`neurox/xbar/ideal.py`, the lossless `--xbar ideal` path via `IdealXbarMacro`) is unrelated and unchanged.

### Moved

- **Modeling schemes live outside the `neurox` core library, in a top-level `works/` package.** The physical 1T1R chip is extracted to `works/offset_1t1r/` (`xbar.py` plus its chip-specific calibration tools, configs, tests, and docs); generic tools (e.g. `xbar_tia`) stay in `neurox/tools/`. `pyproject` packaging now includes `works*`; a scheme self-registers on import, and consumers (examples, tests, tools) `import works.offset_1t1r` before building it through `Xbar.from_config`. `neurox/xbar` exports only the `Xbar` ABC, the `IdealXbar` oracle, and the shared `Core1T1R`.

## Unreleased — Reference Injection + Readout Dissolution

### Changed

- **Reference voltages are injected per call, not self-held (GOAL A).** The ADC family, `VoltageDriver`, and the TIA family no longer carry their reference as a config field / nominal buffer / property; the value is injected per call as a plain `Tensor`. `ADC.convert` gains a keyword-only `v_refs__V: Tensor` (all taps, shape `(*inst, num_refs)`; `adc_mode` indexes its trailing axis, so the mode bound is checked against the tensor, not config), and the abstract `ADC.mode_num` / concrete `available_modes` / `mode_num` are removed — an xbar reports `adc_mode_num` from its reference source's `num_refs`. `VoltageDriver.snapshot` / `TIA.snapshot` gain a keyword `v_ref__V: Tensor` stored in the snap, which the clamp / DC solve reads (the `ClampDriver` role drops its `v_ref__V` member). `GeneralADC` loses `drive_value` / `drive_thermal__V` (and ignores the injected `v_refs__V`); `McsSarAdcConfig` / `SarAdcMonoConfig` lose `v_refs__V`. Consumers own a `VoltageReference` and snapshot it once per forward: the core sources the boundary-clamp reference (BL-clamp + SL-drive taps) once per `cim_read`, and each operating xbar sources the ADC-ladder reference (one tap per mode) once per VMM — one global-scalar draw shared across chunks, preserving chunk bit-exactness.
- **Reference-source taps are now non-negative (GOAL A).** `VoltageReferenceConfig` / `CurrentReferenceConfig` relax tap validation from strictly-positive to non-negative; a `0` V / `0` uA tap denotes a ground/rail reference (relative noise `* 0 == 0`, so it stays stable and exact). This lets the SL driver clamp to ground through the reference source.

### Removed

- **The `ReadOut` container is dissolved (GOAL B).** `neurox/xbar/readout/` is deleted; the offset switch-cap / mux / differential-ADC chain is inlined directly into `Offset1T1RXbar.vec_mat_mul`. `Offset1T1RXbarConfig` drops `readout_config` and gains the flat child configs (`signal_switchcap_config`, `ref_switchcap_config`, `voltage_mux_config`, `adc_config`), the orchestration knobs (`energy_per_op__fJ`, `latency_per_op__ns`), and the owned `adc_v_ref_config`; the policy flattens to `signal_switchcap` / `ref_switchcap` / `voltage_mux` / `bl_adc` / `adc_v_ref`. The readout reference docs are removed and their inbound links repoint to the inline-readout description on the offset xbar page.

## Unreleased — `CurrentReference` / `VoltageReference` Reference Sources

### Added

- **`CurrentReference` / `VoltageReference` analog reference sources** (`neurox/analog/current_reference.py`, `neurox/analog/voltage_reference.py`). Behavioural multi-output reference sources: one module sources a tuple of nominal current (`i_refs__uA`) or voltage (`v_refs__V`) taps, carries the reference's static PPA (area + the always-on bias power folded into `leakage_per_inst__uW`), and hands consumers the actual taps through a `*Snap`, read back through an encapsulated `v_ref__V` / `i_ref__uA` accessor (the source-side `num_refs` property reports how many taps a module sources). They perform no computation and emit no dynamic energy or latency. Two policy-gated non-idealities perturb the taps: a per-die initial-accuracy `tolerance` fixed at fabricate time and a per-read `noise`, both relative (multiplicative). Each is wired into the config/policy/snap modelling system and exported from `neurox.analog`.

## Unreleased — CurrentMirror/CurrentMux Energy Counts Output Side Only

### Changed

- **`CurrentMirror` / `CurrentMux` internal dynamic energy now counts the OUTPUT side only.** `CurrentMirror` drops the input-branch term, logging `v_supply * |i_out| * read_pulse` (was `v_supply * (|i_in| + |i_out|) * read_pulse`); `CurrentMux` already counted output only. Rationale: the input current is sourced externally and its production energy is accounted by the upstream block.

## Unreleased — `AnalogMux` Renamed → `VoltageMux`

### Changed

- **`AnalogMux` renamed `VoltageMux`** (`neurox/analog/analog_mux.py`
  → `neurox/analog/voltage_mux.py`) — the differential
  **voltage**-transport readout leaf becomes the explicit sibling of
  `CurrentMux`, matching the voltage/current split across the analog
  layer. `AnalogMuxConfig` / `AnalogMuxPolicy` → `VoltageMuxConfig` /
  `VoltageMuxPolicy`; the readout composition field `analog_mux_config`
  and policy slot `analog_mux` → `voltage_mux_config` / `voltage_mux`,
  with the matching config / all-off-policy TOML sections renamed.

## Unreleased — Snapshot Dataclasses Renamed `*Snapshot` → `*Snap`

### Changed

- **Per-call snapshot dataclasses renamed `*Snapshot` → `*Snap`** —
  the runtime snapshot dataclasses (`NMOSSnap`, `RRAMSnap`,
  `DriverSnap`, `TIASnap`, `OpAmpTIASnap`, `XbarCellSnap`,
  `XbarCell1T1RSnap`) and the bound type variable (`SnapshotT` →
  `SnapT`) drop the redundant tail so a dataclass no longer reads
  like the `snapshot()` method that produces it. Snapshot-valued
  locals, fields, and parameters move to `snap` / `*_snap` to
  match; the `snapshot()` method name is unchanged.

## Unreleased — `value_range` as Slicer Output + `IdealMacro` Range Tuples

### Changed

- **`Slicer` ABC contract narrowed to 3 keyword-only args** —
  ``value_range`` is no longer a slicer **input**; it is the
  slicer's **output** semantics derived from the slicer's static
  strategy + the three runtime kwargs.  Methods now take exactly
  ``(digit_count, digit_radix, digit_range)``.  This removes the
  awkward placeholder fabrication the mapper had to do for
  slicers that didn't consume the input ``value_range`` anyway.
- **`SerialSlicer` runtime semantics aligned with §8.2** — ``digit_range``
  IS the xbar's primitive input grid; ``digit_radix`` must equal
  ``len(digit_range)``; ``digit_count`` must be ``1``.  All three
  validations live in one private helper and run on every call.
- **`SimpleSlicer` translates the same 3-kwarg call to its outer +
  inner transcoders** — the digit-side envelope check is enforced
  on both ``value_range`` (so the published range matches what
  ``slice`` can actually encode) and ``slice``.
- **`SimpleMapper` drops the `value_range` placeholder** — every
  slicer call threads exactly three kwargs (no fake ``(0, 0)`` or
  ``digit_range``-as-``value_range`` placeholder).  `x_value_range`
  / `w_value_range` forward the same 3-kwarg call to the slicer's
  ``value_range()`` method.
- **`IdealMacro` constructor takes explicit value-range tuples**
  (``x_value_range=...``, ``w_value_range=...``) instead of
  ``w_bits`` / ``x_bits``.  Per §11 of ``docs/dev/architecture/mapping.md``, this
  matches the real ``XbarMacro``'s range surface so the two macro
  families share one operator-facing API.
- **`col_num` / `row_num` reclassified as geometry capability** —
  docs and comments that called them "value-domain capability"
  were wrong; the value-domain quartet is ``x_range``,
  ``w_digit_count``, ``w_digit_radix``, ``w_digit_range``, and
  ``col_num`` / ``row_num`` size the tile, not its value grid.

## Earlier Unreleased — Unified Slicer Contract + Explicit-Arg Call Sites

### Changed

- **`Slicer` ABC unified to a strict 4-kwarg contract** — every
  runtime method (``value_range``, ``slice_radix``, ``slice``) now
  takes exactly ``(value_range, digit_count, digit_radix,
  digit_range)`` as keyword-only arguments; no ``**kwargs``
  variadics anywhere on the slicer surface.  Concrete slicers
  ignore parameters they do not consume (with ``del``) but the
  signature is fixed.  This makes any concrete slicer drop-in
  replaceable from the mapper's point of view — the mapper no
  longer reaches into a slicer's private signature.
- **`SimpleMapper` rewritten as a pure translation layer** — it
  adapts the macro's xbar capabilities to the slicer's unified
  4-kwarg contract and calls every slicer method with the four
  mandatory kwargs spelled out inline.  No ``_x_slicer_kwargs`` /
  ``_w_slicer_kwargs`` helper dicts, no ``**dict`` splat.
- **`XbarMacro` drops ``_xbar_caps()`` and all ``**caps`` splats**
  — ``fabricate`` / ``matmul`` / property getters snapshot the
  xbar's six capabilities to local variables and pass each one by
  explicit keyword argument to every mapper call.  Every cross-
  class call site now documents its full argument list inline.
- **`SimpleSlicer.value_range()` enforces the reachable-domain
  contract** (``docs/dev/architecture/mapping.md`` §9) — if the supplied
  ``digit_range`` cannot cover the signed-digit envelope, the
  strategy is unencodable and ``value_range`` raises rather than
  reporting an inflated range that ``slice()`` would later refuse.
- **`SerialSlicer` rejects inner-digit axes** — passing
  ``digit_count != 1`` now raises; the activation grid carries one
  positional digit per slice and any caller asking for an inner-
  digit axis larger than 1 is using the wrong slicer.

### Removed

- **Self-justifying positional-radix invariant checks** inside
  both slicers (``docs/dev/architecture/mapping.md`` §10) — the slicers no longer
  re-validate that they emitted positional powers of their radix
  immediately after constructing them with the same formula.
  External misuse can't reach those tensors and the assertion
  added only noise.  Any future weighted-reducer migration
  validates compatibility at the consumer, not here.

## Earlier Unreleased — Full-Capability Mapper Contract + Decoupled Tiler Plans

### Added

- **`SignedDigitTranscoder.value_range()`** — the transcoder now
  publishes the symmetric envelope ``(-(r^D - 1), r^D - 1)`` for
  its ``(radix, digit_num)`` triple.  Slicers consume this to
  derive their own ``value_range`` and never hardcode the digit-
  string range formula.
- **Positional-radix invariant check** on both slicers
  (per ``docs/dev/architecture/mapping.md`` §12.3 Plan A).  ``SerialSlicer`` and
  ``SimpleSlicer`` validate that the emitted ``slice_weights`` /
  ``digit_weights`` are exactly the positional powers of the
  relevant radix.
- **`XbarMacro._xbar_caps()` helper** — single private method that
  snapshots the xbar's six primitive capabilities into a plain
  ``dict``.  Every mapper call inside the macro splats this same
  dict, so the macro never decides which subset a mapper consumes.
- **Decoupling smoke test** ``test_simple_mapper_method_ignores_unused_caps``
  asserts varying an irrelevant cap (e.g. ``col_num`` for
  ``x_value_range``) does not change the mapper's output.

### Changed

- **Full-capability kwarg set on every mapper method** — the
  :class:`XbarMapper` ABC now requires every runtime method
  (``x_value_range``, ``w_value_range``, ``x_slice_radix``,
  ``w_slice_radix``, ``map_x``, ``map_w``) to accept the **same**
  six xbar capabilities: ``x_range``, ``col_num``, ``row_num``,
  ``w_digit_count``, ``w_digit_radix``, ``w_digit_range``.  A
  concrete mapper ignores parameters it does not consume; the
  macro never picks a subset.  Decouples the macro from any
  specific mapper's needs.
- **Tiler split into `make_w_plan` / `make_x_plan`** — replaces
  the old single ``make_plan(*, n, k, col_num, row_num)`` with
  two factories: ``make_w_plan(*, n, k, col_num, row_num)`` for
  weight and ``make_x_plan(*, k, row_num)`` for activation.
  ``n = 0`` / ``col_num = 1`` magic values are gone; the two
  paths no longer share placeholder semantics.  ``tile_w`` /
  ``tile_x`` now take their plan as the ``plan=`` keyword
  argument.
- **Slicer value_range delegates to the transcoder**.
  ``SerialSlicer.value_range`` builds a transcoder and reports
  the non-negative half of its envelope (unsigned activation
  grid).  ``SimpleSlicer.value_range`` re-exports the outer
  transcoder's envelope directly.
- **`SimpleMapper` consumes the full kwarg set** and routes to
  ``make_w_plan`` / ``make_x_plan`` separately.

### Removed

- Activation-only ``n = 0`` / ``col_num = 1`` placeholders on the
  tiler.  ``make_w_plan`` rejects ``n < 1`` loudly.

## Unreleased — Tiler / Slicer Split, Unified Mapper, value_range Terminology

### Added

- **`neurox.mapper.xbar.slicer` subpackage** — value-domain
  decomposers with the uniform output contract
  ``[..., slice_num, digit_num]``:
  - :class:`Slicer` ABC + :class:`SlicingResult` dataclass
    (`values`, `slice_weights`, `digit_weights`, `value_range`).
  - :class:`SerialSlicer` — radix-``r`` decomposition with
    structural ``digit_num = 1`` (activation path).
  - :class:`SimpleSlicer` — slice-first-then-digitize, with
    separate ``slice_encoding`` and ``digit_encoding`` knobs
    (weight path).
- **`neurox.mapper.xbar.tiler` subpackage** — matrix tiling:
  - :class:`Tiler` ABC + :class:`TilePlan` dataclass.
  - :class:`SimpleTiler` — right-pad-and-unflatten ``N`` / ``K``;
    operates on tensors carrying the slicer's trailing-2
    ``[slice_num, digit_num]``.
- **`neurox.mapper.xbar.SimpleMapper`** — concrete
  :class:`XbarMapper` composing a shared :class:`Tiler` plus an
  ``x_slicer`` and ``w_slicer``.

### Changed

- **Single unified mapper** — :class:`XbarMapper` ABC replaces the
  ``XbarXMapper`` / ``XbarWMapper`` split.  ``XbarMacro`` now holds
  one ``mapper`` attribute (was ``x_mapper`` + ``w_mapper``); the
  mapper internally composes tiler + activation slicer + weight
  slicer.
- **Range terminology rename**: ``x_levels`` /
  ``w_levels`` → ``x_value_range`` / ``w_value_range`` everywhere
  above the xbar layer (the xbar keeps its primitive ``x_range`` /
  ``w_digit_range`` names).  Updated in
  ``neurox/macro/{base,ideal,xbar_macro}.py``,
  ``neurox/operator/{base,linear,conv2d,spec}.py``,
  ``example/common/macro_factory.py``,
  ``neurox/tools/{xbar_adc_boundaries,analyze_xbar_error_1t1r}.py``,
  and the test suite.
- **Removed**: ``XbarXMapper`` / ``XbarWMapper`` /
  ``SerialXMapper`` / ``SimpleWMapper``.  Migrated to
  ``SerialSlicer`` / ``SimpleSlicer`` / ``SimpleMapper``.
- **Macro factory** builds a :class:`SimpleMapper` with
  :class:`SimpleTiler`, :class:`SerialSlicer`, and
  :class:`SimpleSlicer` from the chip TOML; passes the single
  mapper to :class:`XbarMacro`.

### Tests

- Rewrote ``tests/test_mapper_sign_encoding.py`` (24 tests) for
  the new tiler/slicer/mapper trio: separate unit tests for
  :class:`SerialSlicer`, :class:`SimpleSlicer`,
  :class:`SimpleTiler`, plus integration tests for
  :class:`SimpleMapper`'s ``map_x`` / ``map_w`` shape and
  round-trip behaviour.
- Updated ``tests/test_policy.py`` and ``tests/test_operator.py``
  to use ``w_value_range`` instead of ``w_levels``.

## Unreleased — Mapper Owns Strategy, Macro Dispatches Capabilities

### Changed

- **Mapper owns its static strategy parameters** (per
  ``docs/dev/architecture/config_and_construction.md`` §2.2 / §8).  ``SerialXMapper.__init__`` takes
  ``x_slice_num`` + ``x_encoding``.  ``SimpleWMapper.__init__``
  takes ``w_slice_num`` + ``w_encoding`` + ``w_slice_encoding``.
  Each mapper exposes its strategy params as read-only properties.
- **Mapper methods receive xbar capabilities as separate
  kwargs** — no ``XbarCaps`` bundling.  Each method's signature
  lists exactly the capabilities it consumes:
  - ``XbarXMapper.x_levels(*, x_range)``
  - ``XbarXMapper.x_slice_radix(*, x_range)``
  - ``XbarXMapper.map_x(x, *, x_range, row_num)``
  - ``XbarWMapper.w_levels(*, w_digit_count, w_digit_radix)``
  - ``XbarWMapper.w_slice_radix(*, w_digit_count, w_digit_radix)``
  - ``XbarWMapper.w_digit_num(*, w_digit_count)``
  - ``XbarWMapper.map_w(w, *, col_num, row_num, w_digit_count,
    w_digit_radix, w_digit_range)``
- **`XbarMacro` is a pure orchestrator again** (per
  ``docs/dev/architecture/config_and_construction.md`` §2.3 / §6).  The constructor no longer takes
  ``x_slice_num`` / ``w_slice_num`` / encoding kwargs.  Inside
  ``fabricate`` / ``matmul`` / ``x_levels`` / ``w_levels`` the
  macro reads xbar capabilities into plain locals and threads them
  into the mapper as explicit kwargs.
- **`XbarCaps` dataclass removed.**  Per user direction
  capabilities flow as separate kwargs at each call site —
  no bundling.
- **`example/common/macro_factory.py`** passes strategy params
  into the mapper constructors directly; the macro call site is
  pure composition.

### Tests

- Rewrote ``tests/test_mapper_sign_encoding.py`` (17 tests) for
  the mapper-owns-strategy API: mappers built with strategy
  kwargs at construction; every call site lists xbar capability
  kwargs inline.  Envelope-validation failure now flows through
  ``map_w`` (the method that actually consumes ``w_digit_range``).

## Unreleased — Pure-Tool Mappers, Macro as Capability Dispatcher

### Changed

- **Mappers are now stateless pure tools** (per ``docs/dev/architecture/config_and_construction.md``
  §1.2 / §2 / §5 / §8 / §9).  ``SerialXMapper`` and
  ``SimpleWMapper`` are constructed with no arguments.  They hold
  no buffers, no xbar reference, no capability snapshot, and no
  pre-built transcoders.  Every method (``x_levels`` /
  ``x_slice_radix`` / ``map_x`` and ``w_levels`` /
  ``w_slice_radix`` / ``map_w``) receives the xbar primitive
  capabilities (``XbarCaps``) and the macro strategy parameters
  (``x_slice_num`` / ``w_slice_num`` / encoding choices) as
  explicit keyword arguments at call time.  Internal transcoders
  are built per call.
- **`XbarMacro` becomes the strategy owner and capability
  dispatcher** (per ``docs/dev/architecture/config_and_construction.md`` §1.3 / §3 / §6).  The
  constructor now takes the strategy parameters directly:
  ``x_slice_num``, ``w_slice_num``, ``x_encoding``, ``w_encoding``,
  ``w_slice_encoding``.  Inside ``fabricate`` / ``matmul`` /
  ``x_levels`` / ``w_levels`` the macro reads its xbar's primitive
  capabilities into a fresh ``XbarCaps`` and dispatches caps +
  strategy to the (stateless) mapper.  Shift-adder radixes are
  queried from the mappers once per matmul call and passed into
  the compiled ``_digital_aggregate`` as Python ints.
- **New `XbarCaps` dataclass** (``neurox.mapper.xbar.XbarCaps``).
  Frozen bundle of the six value-domain capabilities the mapper
  needs: ``x_range``, ``col_num``, ``row_num``, ``w_digit_count``,
  ``w_digit_radix``, ``w_digit_range``.  Exported from
  ``neurox.mapper.xbar``.
- **Abstract bases** ``XbarXMapper`` / ``XbarWMapper`` are now
  runtime-driven: every accessor is a method that takes ``caps`` +
  strategy kwargs.  The ``w_digit_num`` property is gone — use
  ``caps.w_digit_count``.
- **`example/common/macro_factory.py`** builds the mappers with
  zero-argument constructors (``SerialXMapper()`` /
  ``SimpleWMapper()``) and passes the strategy knobs straight
  into ``XbarMacro``.

### Tests

- Rewrote ``tests/test_mapper_sign_encoding.py`` (18 tests) for
  the pure-tool mapper API: helper ``_caps(...)`` produces a
  literal ``XbarCaps`` (no IdealXbar dependency); every call site
  threads ``caps`` + strategy through method kwargs; encoding-
  changes-shape-identical and multi-slice round-trip checks moved
  to method-arg style.

## Unreleased — Mappers Own Their Strategy (Re-correction)

### Changed

- **`XbarMacro` is a pure orchestrator** (per ``docs/dev/architecture/config_and_construction.md`` §2).
  The constructor goes back to accepting ``x_mapper`` and
  ``w_mapper`` as already-built instances; the
  ``sa_num / sw_num / x_encoding / w_encoding / slice_encoding``
  parameters are gone, as is the internal transcoder / mapper
  construction.  Macro keeps a small composition-time check
  (``w_mapper.w_digit_num == xbar.w_digit_count``) but does not
  hold any mapping policy.
- **`SerialXMapper` owns its own strategy** — the constructor now
  takes ``xbar``, ``x_slice_num``, ``x_encoding``; reads
  ``x_range`` / ``col_num`` / ``row_num`` from the xbar; builds
  the activation transcoder internally.
- **`SimpleWMapper` owns its own strategy** — the constructor now
  takes ``xbar``, ``w_slice_num``, ``w_encoding``,
  ``w_slice_encoding``; reads every relevant capability from the
  xbar; builds the two internal transcoders.  The old
  ``w_transcoder`` / ``col_num`` / ``row_num`` / ``w_digit_count``
  / ``w_digit_radix`` / ``w_digit_range`` / ``sw_num`` /
  ``slice_encoding`` kwargs are gone.
- **Variable rename to non-shorthand names** (per
  ``docs/dev/architecture/config_and_construction.md`` §4): ``sa_num`` → ``x_slice_num``,
  ``sa_radix`` → ``x_slice_radix``, ``sw_num`` → ``w_slice_num``,
  ``sw_radix`` → ``w_slice_radix``.  ``Sa / Sw / Tc / Tr / M``
  remain only as shape-annotation shorthand.  Tile counts use
  ``col_tile_num`` / ``row_tile_num`` as variable names.
- **`WMappingResult` field rename**: ``tile_in_dim`` →
  ``col_tile_num``; ``tile_out_dim`` → ``row_tile_num``.
- **`example/common/macro_factory.py`** builds both mappers
  directly from TOML knobs (``x_transcoder.digit_num``,
  ``x_transcoder.encoding``, ``w_transcoder.encoding``,
  ``w_transcoder.w_slice_num`` defaulting to 1,
  ``w_transcoder.w_slice_encoding`` defaulting to ``true_form``)
  and passes them to the macro.

### Tests

- Rewrote ``tests/test_mapper_sign_encoding.py`` (18 tests) for
  the new mapper API: helper ``_xbar(...)`` builds a minimal
  :class:`IdealXbar` so the mappers can read primitive
  capabilities at construction; assertions use the new
  ``x_slice_num`` / ``w_slice_num`` / ``x_slice_radix`` /
  ``w_slice_radix`` / ``row_tile_num`` / ``col_tile_num`` names.

## Unreleased — Macro Owns Mapping Strategy

### Changed

- **`XbarMacro.__init__` now owns the mapping strategy parameters
  directly** (per ``docs/dev/architecture/config_and_construction.md`` §7).  The constructor takes
  ``sa_num``, ``sw_num``, ``x_encoding``, ``w_encoding``,
  ``slice_encoding`` and builds the underlying transcoders +
  ``SerialXMapper`` + ``SimpleWMapper`` internally.  The macro is
  the source of truth for ``Sa`` / ``Sw`` / encoding choices;
  mappers stay pure tools that carry no hidden policy.
- **`SimpleWMapper` accepts `slice_encoding` explicitly** (per
  ``docs/dev/architecture/config_and_construction.md`` §8).  The slicer's encoding is no longer the
  hardcoded ``"true_form"``; it now arrives from the macro and is
  forwarded into the slicer's :class:`SignedDigitTranscoder`.
- **`example/common/macro_factory.py`** reads strategy parameters
  from the chip TOML and forwards them to ``XbarMacro``;
  transcoders + mappers are no longer constructed at the factory
  level.

### Docs

- Generic xbar docs stop calling ``[Sa, Sw, row_num]`` and
  ``[Sa, Sw, data_num]`` "primitive core tails" — they're macro-
  side full layouts, not xbar primitive contracts.  The xbar
  primitive sees only the trailing ``[row_num]`` / ``[data_num]``.
- ``XbarMacro``'s class docstring rewrites the tiling + digital-
  aggregation discussion accordingly.
- ``IdealXbar.vec_mat_mul`` and ``Offset1T1RXbar.vec_mat_mul``
  docstrings now state the generic primitive contract first and
  flag any internal data-broadcast ``unsqueeze`` as an
  implementation detail.
- Removed the "``Sw = 1`` placeholder today" language; documented
  that ``SimpleWMapper`` supports any ``sw_num >= 1`` and that the
  remaining knob is the slicing-encoding policy, not the existence
  of the ``Sw`` axis.
- ``neurox/mapper/transcoder.py`` module docstring refreshed to
  reference the split mapper classes (``XbarXMapper`` /
  ``XbarWMapper``) instead of the removed ``XbarMapper``.

### Tests

- Updated ``tests/test_mapper_sign_encoding.py``: every
  ``SimpleWMapper(...)`` call passes the new required
  ``slice_encoding`` keyword; added
  ``test_simple_slice_encoding_threads_through`` to assert the
  encoding actually reaches the internal slicer.  18/18 mapper
  tests pass.

## Unreleased — Shape / Doc Cleanup + Multi-Slice SimpleWMapper

### Changed

- **`neurox/xbar/base.py`** — generic xbar docs now publish only the
  primitive shape contract (``w[..., data_num, digit_num, row_num]``,
  ``x[..., row_num]``, ``y[..., data_num]``).  Macro-side axes
  (``Bw``, ``Bx``, ``M``, ``Tc``, ``Tr``, ``Sa``, ``Sw``) are
  documented as belonging to the macro layer; the dim-symbol table
  is removed.  Row / column terminology is topology-agnostic (cells
  sharing one input vs. cells aggregating to one output); the
  ``XbarConfig`` docs no longer bind those to WL / BL / SL.  The
  "trivially derivable w_range" claim is dropped — that range is
  encoding-dependent and lives in the mapper.
- **`neurox/xbar/_1t1r/offset_1t1r.py`** — replaced stale
  ``neurox.mapper.XbarMapper`` references with the new split mapper
  classes; removed the stale "(y, dynamic_energy__fJ)" return
  wording.  ``vec_mat_mul`` docstring now states the generic
  primitive contract first and presents the internal
  ``x.unsqueeze(-2)`` as an implementation-specific detail.
- **`neurox/xbar/_1t1r/core_1t1r.py`** — ``Core1T1R.forward``
  describes only the shape it actually consumes
  (``broadcastable to [..., 1, row_num]``); no macro-level
  ``Sa, Sw`` terminology.  Standardised ``# Shape: …`` comments at
  each transformation (broadcast expand, snapshots, DAC convert,
  solver result, post-solver clamp).
- **`SimpleWMapper` supports caller-specified ``Sw``**: the
  ``sw_num != 1`` restriction is gone.  The mapper now uses an
  internal ``SignedDigitTranscoder("true_form", radix=r^D,
  digit_num=sw_num)`` to decompose the algorithm weight into
  ``sw_num`` per-xbar-word slices, then digitizes each slice with
  the supplied ``w_transcoder``.  ``w_levels`` widens to
  ``(-(sw_radix^sw_num - 1), sw_radix^sw_num - 1)`` with
  ``sw_radix = r^D``.

### Tests

- Added multi-slice cases to ``tests/test_mapper_sign_encoding.py``:
  ``Sw=2`` and ``Sw=3`` level derivation, ``map_w`` shape +
  ``slice_weights``, and a full slice → digitize → decode →
  positional-recombine round-trip that confirms the algorithm-side
  value survives the mapping.
- Replaced the stale ``test_simple_sw_num_gt_1_not_implemented`` with
  a positive guard ``test_simple_sw_num_zero_rejected``.

## Unreleased — Shape Contract Refinement

### Changed

- **Unified shape contract** per ``docs/dev/architecture/mapping.md``:
  - Weight: ``[Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, digit_num, row_num]``.
    ``M = 1`` now sits BEFORE ``Tc`` (previously was between ``Tc``
    and ``Tr``); the result is a clean canonical leading
    ``[Bw, M=1, Tc, Tr, Sa=1, Sw]``.
  - Activation: ``[Bx, M, Tc, Tr=1, Sa, Sw=1, row_num]``.  The
    historical ``row = 1`` placeholder is dropped from the
    primitive core tail.
- **Drop `row_size` / `col_size` aliases** in the mappers; use
  ``data_num`` (= xbar's ``col_num`` for offset 1T1R) and
  ``row_num`` directly to match the spec's symbol table.
- **`Xbar.vec_mat_mul` is now abstract** — subclasses
  (``FakeXbar``, ``Offset1T1RXbar``) implement it directly.  The
  base no longer has a thin wrapper around an internal
  ``_vec_mat_mul_impl``; that hidden hook is removed.
- **Subclasses absorb the dropped row=1 axis internally** —
  ``FakeXbar.vec_mat_mul`` does ``x.unsqueeze(-2)`` to add the data
  broadcast slot; ``Offset1T1RXbar.vec_mat_mul`` does the same
  before calling ``Core1T1R``.  Callers see only the
  ``[Sa, Sw, row_num]`` primitive core tail.

### Tests

- Updated ``tests/test_mapper_sign_encoding.py``: ``x_xbar`` is now
  6-D for a 2-D input (the ``row = 1`` axis is gone); shape
  assertions reflect the new canonical layout.

## Unreleased — Macro / Mapper / Transcoder Layer Boundaries

### Added

- **Xbar mapper subpackage** (`neurox.mapper.xbar`) with one file
  per class:
  - ``base.py`` — abstract bases ``XbarXMapper`` (activation) and
    ``XbarWMapper`` (weight); result dataclasses ``XMappingResult``,
    ``WMappingResult``.
  - ``serial_x_mapper.py`` — concrete ``SerialXMapper``: radix-digit
    decompose + K tile.
  - ``simple_w_mapper.py`` — concrete ``SimpleWMapper``: slice-first-
    then-digitize at ``sw_num = 1``.  The slicing step is explicit
    even at one slice; multi-slice extension only replaces
    ``_slice``.
- **`Transcoder.radix`** abstract property on the
  :class:`neurox.mapper.Transcoder` base.

### Changed

- **`XbarMacro.__init__`** takes two separate mapper instances —
  ``x_mapper: XbarXMapper`` and ``w_mapper: XbarWMapper`` — instead
  of a unified mapper.  All components (``xbar``, ``x_mapper``,
  ``w_mapper``, ``col_accumulator``, ``w_shift_adder``,
  ``x_shift_adder``, ``requantizer``) are instance-based (no
  factories, no internal child-name plumbing).  The macro accepts
  no ``name=``.
- **`XbarMacro.{w_levels, x_levels}`** delegate to the respective
  mapper.  The mapper is the source of truth; the macro is the
  Protocol-facing surface.
- **`XbarMacro._digital_aggregate`** reads ``x_mapper.sa_radix`` and
  ``w_mapper.sw_radix`` for the two shift-adder reductions.
- **`example/common/macro_factory.py`** builds both mappers
  (``SerialXMapper`` and ``SimpleWMapper``) plus every digital child
  with its hierarchical profiler name set, and passes fully-built
  instances to ``XbarMacro``.
- **`neurox.mapper.__init__`** now exposes only the shared digit
  encoder (``Transcoder``, ``SignedDigitTranscoder``, ``Encoding``).
  Xbar-macro mappers are imported from the subpackage directly:
  ``from neurox.mapper.xbar import XbarXMapper, XbarWMapper, ...``.

### Removed

- **`Xbar.w_range`** — encoding-dependent value range is *not* a
  primitive xbar capability and belongs in the weight mapper.
  Removed from the abstract base.
- **`XbarMapper`** and **`InternalDigitOnlyMapper`** — the unified
  mapper structure is wrong (activation and weight pipelines are
  different problems) and `InternalDigitOnlyMapper`'s "tile-then-
  digitize" approach conflates xbar-internal digit grids with macro-
  external slicing.  Replaced by the activation/weight split.
- **`neurox.mapper.XbarMapperConfig`** (empty placeholder dataclass).

### Tests

- **Rewrote** ``tests/test_mapper_sign_encoding.py`` (14 tests) for
  the activation/weight mapper split: serializing, level derivation,
  slice-first-then-digitize pipeline, full transcoder ↔ xbar
  contract validation, ``sw_num > 1`` rejection, padding, decode
  round-trip.

## Unreleased — Top-level API + Profiler Refactor

### Added — Stage 1 / Stage 2 / Stage 3 surface

- **Top-level `neurox` package surface**: ``import neurox`` now exposes
  every public entry point: ``build_evaluator``, ``replace_model``,
  ``load_neurox_state``, ``bind_output_calibration``, ``fabricate_model``,
  ``ReplacementPolicy``, ``ReplacementRule``, ``ReplacementContext``,
  ``default_policy``, ``name_excluded_match``, ``by_attr_match``,
  ``heterogeneous_macro_policy``, ``NeuroxStateError``, ``StateBindingReport``,
  ``StructuralReport``, ``NeuroxProfiler``, ``ProfilerReport``,
  ``StaticMetrics``, ``RuntimeEvent``, ``StaticRecord``, ``ProfiledModule``,
  ``QuantSpec``, ``extract_neurox_state``, ``replace_for_hat``,
  ``freeze_hat_observers``, ``fold_batchnorm``.
- **Staged ``build_evaluator``**: four new public functions
  (``replace_model``, ``load_neurox_state``, ``bind_output_calibration``,
  ``fabricate_model``); the one-shot wrapper is now a thin
  orchestrator over them.
- **Rule-driven replacement policy** (`neurox.replace.policy`):
  ``ReplacementRule`` + ``ReplacementPolicy`` with priority ordering,
  ``unmatched={"keep_float", "warn", "error"}``, helper constructors
  ``name_excluded_match``, ``by_attr_match``,
  ``heterogeneous_macro_policy``.  Default behaviour preserved by
  ``default_policy(macro_factory)``.
- **Strict checkpoint validation**: ``load_neurox_state`` raises
  ``NeuroxStateError`` when required NeuroX operator buffers are
  missing.  ``StateBindingReport`` carries the loaded / missing /
  unexpected diagnostic for non-strict callers.
- **Side-channel profiler** (`neurox.profiler.ProfiledModule`):
  hierarchical names propagate through every physical leaf;
  ``NeuroxProfiler.analyze_static`` aggregates area + leakage power
  from ``_inst_area__um2`` / ``_inst_leakage__uW`` totals stored on
  each module at fabricate time.  Leakage *energy* is derived
  centrally as ``leakage_power__uW * total_latency__ns`` in
  ``NeuroxProfiler.summary``.
- ``extract_neurox_state(model)`` — clean name for the
  HAT-state-extraction helper (NeuroX-owned checkpoint boundary).

### Changed — physical-module signatures

- Numerical functions now return clean numerical results.  Dynamic
  energy flows exclusively through the profiler side channel.
  Affected signatures:
  - ``DAC.convert(code) -> Tensor`` (was ``(Tensor, Tensor)``)
  - ``SwitchCap.sample_and_accumulate(v) -> Tensor``
  - ``AnalogMux.transport(v_pos, v_neg) -> (v_pos, v_neg)`` (no energy)
  - ``ADC.convert(...) -> Tensor`` (every concrete ADC)
  - ``Accumulator.operate / ShiftAdder.operate / Requantizer.operate -> Tensor``
  - ``Xbar.vec_mat_mul(x) -> Tensor``
  - ``XbarMacro.matmul(...) -> Tensor`` (and ``FakeMacro.matmul``)
  - ``run_matmul_pipeline(...) -> Tensor``
  - ``Core1T1ROutput`` and ``ReadOutOutput`` drop their
    ``dynamic_energy__fJ`` fields (plus per-leg energy fields on
    ``ReadOutOutput``).
- Every physical-module ``__init__`` accepts ``name: str`` for
  hierarchical profiler identity.  Composites
  (``Core1T1R``, ``OpAmpTIA``, ``OffsetSwitchCapMuxAdcReadOut``,
  ``Offset1T1RXbar``, ``XbarMacro``) thread child names from their
  own.  Macro factories are now ``Callable[[str], NeuroxMacroQuantMatMul]``.
- ``XbarMacro`` no longer exposes ``area__um2``,
  ``leakage_power__uW``, ``leakage_energy__fJ``, or ``latency__ns``;
  static aggregation lives on the profiler instead.  The
  ``NeuroxMacroQuantMatMul`` Protocol drops the corresponding
  required properties.
- ``QuantLinear.fabricate`` / ``QuantConv2d.fabricate`` no longer
  call ``NeuroxProfiler.log_area``; ``QuantLinear.forward`` /
  ``QuantConv2d.forward`` no longer call ``NeuroxProfiler.log_energy``.
- ``torch._dynamo.config.suppress_errors = True`` is enabled at
  ``import neurox`` time so dynamo gracefully degrades when it hits
  the ``@torch.compiler.disable``'d side-channel call site.

### Deprecated

- ``neurox.replace.fabricate`` is a deprecation alias for
  ``fabricate_model``.
- ``neurox.replace.extract_hat_state`` is a deprecation alias for
  ``extract_neurox_state``.
- Both aliases will be removed after one release cycle.

### Tests

- **New**: ``test_profiler.py`` (14), ``test_build_evaluator.py``
  (7), ``test_policy.py`` (10), ``test_top_api.py`` (9) — 40 new
  tests covering Phase A/B/C/D.
- **Modified**: ``test_signal_chain.py`` updated to the new
  no-tuple physical-module signatures.

## Unreleased — ADC, TIA, Decoder, Mapper Redesign

### Added

- **Physics-based ADC family** (`neurox.analog.adc`): concrete
  topologies with per-tech-node energy / latency models — `GeneralADC`
  (boundary-bucketize fallback), `McsSarAdc`, `SarAdcMono`,
  `PipelineADC`, `CyclicADC`, `RampADC`.  All share an `ADC` ABC and a
  multi-mode `(n_bits, n_states, max_signal)` configuration via
  `ADCMode`.
- **Transimpedance amplifier** (`neurox.analog.tia.TIA`): owns the BL
  clamp voltage and the column-current → voltage stage.  State-
  dependent latency follows the NeuroSIM `CurrentSenseAmp.cpp`
  polynomial fit.
- **Analog multiplexer** (`neurox.analog.mux.AnalogMux`): models
  `numColMuxed` column sharing with explicit PPA accounting.
- **Decoder** (`neurox.analog.decoder.Decoder`): wraps the WL DAC,
  owns row decoding and optional bit-serial expansion.
- **Stochastic rounding** in every quantizer
  (`neurox.common.quant.stochastic_floor_div`,
  `stochastic_floor_to_int`, `floor_bucketize`).  Auto-gated on
  `module.training`; `RequantizerConfig.stochastic` and per-ADC-config
  `stochastic` flags accept manual overrides.
- **Sign + value encoding strategies** in `XbarMapper`:
  `sign_encoding ∈ {"differential", "twosided", "offset", "native"}`
  and `value_encoding ∈ {"sign_magnitude", "two_sided_balanced"}`.
  Validates compatibility against the xbar's `signed_capability`.
- **`Xbar.fabricate_unmapped(w)`** convenience for direct callers
  (tests, tools).
- **`Xbar.ideal_states_per_column()`** abstract method enabling the
  macro to derive `output_rescale_factor` analytically.
- **Calibration tool overhaul** (`neurox.tools.xbar_adc_boundaries`,
  renamed from the misspelled `xbar_adc_boundries`): two-fold
  range + precision output via real-vs-ideal xbar comparison.  CLI
  flags `--random`, `--noise`, `--visualize`, `--output`.

### Changed

- **Sign-split moved from xbar into mapper.**  `Xbar1T1R.fabricate`
  and `XbarIdeal.fabricate` no longer call `_w_sign_split`; the
  mapper's `map_w` produces sign-encoded tensors directly.
- **`output_rescale_factor` is now derived** by the macro from
  `xbar.ideal_states_per_column()` and the bound ADC's `n_states()`.
  The TOML field is now optional (`float | None`); existing configs
  with a numeric override continue to work.
- **`floor_boundaries_for_mode`** replaces the round-style
  `(c - 0.5) · LSB` placement with floor-style `c · LSB` placement.
  ADC family members use `right=True` bucketize for proper floor
  semantics at exact boundaries.
- **`XbarIdeal._vec_mat_mul_impl`** uses `stochastic_floor_to_int`
  instead of `torch.round` — unbiased under coarse output grids;
  `module.eval()` falls back to deterministic floor.
- **Default `default_1t1r.toml`** drops the explicit
  `output_rescale_factor` (auto-derived) and adds an `IDEAL_1T1R_TOML`
  fixture for tests.

### Removed

- `Xbar1T1R.fabricate` no longer expects raw `[col_num, row_num]`
  input — it expects the mapper's sign-encoded output.  Use
  `xbar.fabricate_unmapped(w)` for the legacy contract.

### Tests

- **New**: `test_adc_family.py`, `test_signal_chain.py`,
  `test_stochastic_rounding.py`, `test_mapper_sign_encoding.py`
  (38 new tests covering the redesign primitives).
- **Modified**: `test_xbar_macro.py`, `test_xbar_physics.py` —
  direct fabricate callers updated to use `fabricate_unmapped`;
  manual `_x_sign_split` / `_w_sign_split` calls removed where the
  mapper now handles them.

### Documentation

- New design docs:
  - [`adc_models.md`](doc/design/adc_models.md)
  - [`signal_chain.md`](doc/design/signal_chain.md)
  - [`mapping_strategies.md`](doc/design/mapping_strategies.md)
  - [`calibration.md`](doc/design/calibration.md)
