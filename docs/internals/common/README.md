# Common

The cross-cutting software primitives shared across every NeuroX subsystem, mirroring the `neurox/common/` package.

- [encoding/](encoding/README.md) — the `Transcoder` codec layer between integers and signed-digit strings.
- [nonideality](../primitive/nonideality.md) — the reusable analog non-ideality kernels and their configs.
- [physical_constant](../primitive/physical_constant.md) — the canonical SI physical constants and the derived thermal voltage.
- [prober](prober.md) — the `Prober` collector for the probe side channel: per-call tensor capture on named channels.
- [profiler](profiler.md) — the `NeuroxProfiler` collector for the PPA side channel.
- [quant](quant.md) — the shared quantization kernels: stochastic-rounding and fake-quantize.
- [serialize](../../api/configuration.md) — the dataclass serialization machinery: config and policy file loading, directives, and presets.
