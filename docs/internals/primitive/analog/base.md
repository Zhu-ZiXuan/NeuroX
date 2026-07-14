# Analog base

## Summary

`neurox/primitive/analog/base.py` defines the analog family's local triple: `AnalogConfig` and `AnalogPolicy` — empty markers extending [`ConfigBase` / `PolicyBase`](../../common/base.md) — and `AnalogBase`, a `ModuleBase[ConfigT, PolicyT]`-only base that adds no `ProfileMixin` and no PPA. A sized analog leaf composes `ProfileMixin` and declares its own area / leakage on top of `AnalogBase`; the two bare leaves (current mirror, current mux) stay `AnalogBase`-only.

## Design decisions

- **`AnalogBase` carries no PPA.** The base is `ModuleBase` only — it binds no `ProfileMixin` and declares no area / leakage. Whether an analog block reports static silicon is a per-leaf decision, not a family-base one: a sized leaf adds `ProfileMixin` and sets the bare per-instance PPA data it aggregates into `area__um2` / `leakage__uW`. This is what lets the ideal current mirror and current mux — which own no reportable silicon and emit nothing — stay bare `AnalogBase` leaves while the driver, references, voltage mux, switch-cap bank, and the ADC / DAC / TIA families are sized.
- **Empty markers `AnalogConfig` / `AnalogPolicy`.** The family shares no field across its leaves, so each carries its own `*Config` / `*Policy` subclass; the markers exist only to give the family a named config / policy base type.
- **Sized-leaf pattern.** A sized leaf sets `_area_per_inst__um2` / `_leakage_per_inst__uW` in `__init__` (usually from config), and `ProfileMixin` aggregates each as `× inst_count` into `area__um2` / `leakage__uW`; the two per-instance config fields (`area_per_inst__um2`, `leakage_per_inst__uW`) live on the leaf's own config, or on the family config base for the ADC / DAC / TIA families. The current reference is sized — its always-on bias power rides `leakage_per_inst__uW`.
- **The bare mirror / mux emit nothing.** The current mirror logs no energy, and the current mux applies only its matched gain and self-logs no latency, so neither composes `ProfileMixin`; their static cost rolls up into the owning circuit's config.

## Contracts & invariants

- **`AnalogBase` adds no `__init__`.** It inherits `ModuleBase.__init__(*, config, policy, name="", inst_shape)`; a leaf's own `__init__` additionally takes `dtype` / `T__K`, forwards `config` / `policy` / `name` / `inst_shape` up, and builds its buffers.
- **Sized vs bare membership.** Sized (`ProfileMixin` plus own area / leakage): `VoltageDriver`, `VoltageReference`, `CurrentReference`, `VoltageMux`, `SwitchCap`, and the `ADC` / `DAC` / `TIA` family bases. Bare (`AnalogBase`-only, no PPA): `CurrentMirror`, `CurrentMux`.

---

- **Reference**: N/A — software base; per-leaf area / leakage numbers are specified under [reference](../../../reference/primitive/analog/README.md)
- **Implementation**: `neurox/primitive/analog/base.py`
- **Tests**: TODO — covered indirectly via the profiler's static collection
