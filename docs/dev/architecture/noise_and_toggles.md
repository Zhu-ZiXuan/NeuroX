# Noise and Mismatch Configuration

This document defines how non-ideality and mismatch sources are configured and gated across NeuroX.

## Rule

For every non-ideality / mismatch / dynamic-noise source the project models:

1. The cfg holds **parameter values** as fully-populated, non-Optional fields. Scalar sigmas are required `float`; multi-parameter distributions are required sub-config dataclasses.
2. Each source is paired with a required `enable_<source>: bool` toggle on the same cfg level. The toggle has **no default** — every chip TOML must declare it explicitly.
3. `None` is forbidden in any cfg field that carries a noise / mismatch parameter. `Optional[...]` is not used for these.
4. Runtime helpers in `neurox/common/nonideality.py` take a `*, enabled: bool` kw-only parameter and short-circuit to a pass-through when `enabled=False`. Callers therefore write a single unbranched expression — no `if`-gates at the call site.

## Why

The cfg conflates two distinct facts when a single `Optional[float]` field is used: "what the underlying physical value is" and "whether we apply it in this run". Splitting them into `(value, enable_value)` makes both visible:

- The value stays honest to the PDK / device-foundry datum even when the simulation disables the source.
- Experimental ablations toggle the boolean instead of overwriting parameter values, so cfg diffs cleanly show intent.
- Sigmas that are *derived* at runtime from other quantities (e.g. Pelgrom area scaling) are not forced into a misleading `None` to disable; the toggle stays purely declarative.
- All `apply_*` helpers become pure math gated by one uniform mechanism.

## Naming

- Parameter field: `<source>__<unit>` when a physical unit exists, otherwise `<source>` (relative / probability / shape names).
- Toggle: `enable_<source>` matching the parameter field's stem. The verb form is consistent project-wide; do not use alternatives like `<source>_enabled` or `<source>_active`.

When a category sums multiple parameter fields, the toggle name reflects the *category*, not any one parameter. Example: kT/C sampling noise in MCS-SAR and SwitchCap is gated by a single `enable_sampling_thermal_noise: bool` even though the sigma is derived from per-instance capacitance.

## Field grouping

Place each `enable_<source>` directly below the parameter (or sub-config) it gates. Group related fields with the `# --- Group name ---` header convention (see [`code_style.md`](code_style.md#step-comments)) — one blank line above, one blank line below, no trailing `#`. The same grouping applies to TOML config files: place toggles next to their values and use the identical comment style at the TOML level. A cfg block that mixes noise sources, design knobs, and PPA fields should read as a sequence of small labelled groups, not a flat field list.

Example:

```python
@dataclass(frozen=True)
class FooConfig(ValidateMixin):
    # --- Drive thermal noise ---
    drive_thermal__V: float
    enable_drive_thermal: bool

    # --- PPA ---
    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
```

## Where parameter values live

- **Device-level PDK facts** (Pelgrom coefficients, intrinsic device noise sigmas, sub-config parameter blocks) live in `neurox/config/process/<class>.toml` alongside the other physical parameters of that device variant. These files stay portable across project boundaries and contain **no `enable_*` toggles**.
- **Toggles** live one layer up, on the consuming chip / macro / circuit cfg. In `default_1t1r.toml` they appear inline alongside the `_neurox_use` directive that pulls in the device's physical parameters.

This split keeps `process/*.toml` as pure PDK records and concentrates experimental-decision state in the chip-level TOML.

## Runtime semantics

`apply_*` helpers in `nonideality.py` follow a uniform shape:

```python
def apply_<noise>(x: Tensor, ...args..., *, enabled: bool) -> Tensor:
    if not enabled:
        return x
    ...
```

Callers therefore write a single line per source:

```python
g__uS = apply_state_dependent_gamma(g__uS, cfg.prog_gamma, enabled=cfg.enable_prog_gamma)
g__uS = apply_telegraph_noise(g__uS, cfg.read_telegraph, enabled=cfg.enable_read_telegraph)
g__uS = apply_gaussian(g__uS, cfg.read_thermal__uS, enabled=cfg.enable_read_thermal)
```

No `if` branches at the call site, no `None` checks inside the helpers, no special-case dispatch.

Static-mismatch `apply_*` calls live inside `_sample_fabricate_mismatch()` (the `FabricateMixin` override point); dynamic per-call noise lives inside `convert` / `snapshot` / similar runtime methods. The cfg toggles are read directly there; the cadence at which the surrounding `fabricate()` is invoked is the operator-level concern documented in [`fabrication_lifecycle.md`](fabrication_lifecycle.md).

## What this rule does **not** apply to

- Numerical hyperparameters that are not noise (e.g. softclip softness, learning rates). These follow the regular "required field" rule but do not need a paired toggle.
- Boolean construction modes that already act as toggles (e.g. `bit_serial: bool` on the decoder, `input_transform: Literal["linear", "log2"]`). These are structural choices, not noise.
- The training-mode flag (`self.training`) carried by `nn.Module` itself; that is runtime, not cfg. Stochastic-rounding kernels in `neurox/common/quant.py` consume it directly — there is no separate `stochastic` override knob.

## Adding a new noise source

1. Add the parameter field (scalar `float` or sub-config dataclass) to the cfg, no default.
2. Add the paired `enable_<source>: bool` field, no default.
3. Add a `validate_*` clause for the parameter (`_require_nonneg`, `_require_pos`, …).
4. In the runtime, call the matching `apply_*` helper with `enabled=cfg.enable_<source>`.
5. Add the parameter value to the relevant `process/*.toml` if it is a PDK fact, otherwise to the chip TOML.
6. Add the toggle to every chip TOML that consumes the cfg.
