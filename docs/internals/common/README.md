# Common

The cross-cutting software primitives shared across every NeuroX subsystem, mirroring the `neurox/common/` package.

- [circuit](circuit.md) — the fixed `CircuitBase` and its PPA-carrying config base.
- [encoding/](encoding/README.md) — the `Transcoder` codec layer between integers and signed-digit strings.
- [load_dump](load_dump.md) — the sole TOML / YAML format boundary for the config dataclass tree.
- [nonideality](nonideality.md) — the reusable analog non-ideality kernels and their configs.
- [physical_constant](physical_constant.md) — the canonical SI physical constants and the derived thermal voltage.
- [profiler](profiler.md) — the `NeuroxProfiler` collector for the PPA side channel.
- [quant](quant.md) — the shared stochastic-rounding quantization kernels.
- [mixin/](mixin/README.md) — the reusable mixin classes for circuit modules and configs.
