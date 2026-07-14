# Common

The cross-cutting software primitives shared across every NeuroX subsystem, mirroring the `neurox/common/` package.

- [base](base.md) — the `ModuleBase` module root and the `ConfigBase` / `PolicyBase` config-dataclass roots.
- [encoding/](encoding/README.md) — the `Transcoder` codec layer between integers and signed-digit strings.
- [nonideality](../primitive/nonideality.md) — the reusable analog non-ideality kernels and their configs.
- [physical_constant](../primitive/physical_constant.md) — the canonical SI physical constants and the derived thermal voltage.
- [profiler](profiler.md) — the `NeuroxProfiler` collector for the PPA side channel.
- [quant](quant.md) — the shared quantization kernels: stochastic-rounding and fake-quantize.
- [serialize/](serialize/README.md) — the dataclass serialization machinery: dict <-> file I/O, coercion, `_neurox_class` dispatch, `_neurox_use` / `_neurox_use_preset` directives.
- [mixin/](mixin/README.md) — the reusable mixin classes for circuit modules and configs, including the `SerializeMixin` front onto `serialize/`.
