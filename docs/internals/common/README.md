# common — Implementation

The cross-cutting software primitives shared across every NeuroX subsystem: the electrical-circuit base, config serialization, non-ideality kernels, physical constants, profiling, quantization, and the reusable mixins.

- [circuit](circuit.md) — the typed electrical-circuit base and its PPA-carrying config base.
- [encoding/](encoding/README.md) — the `Transcoder` ABC, registry-backed encoding dispatch, and the per-encoding signed-digit codec contracts.
- [load_dump](load_dump.md) — dataclass serialization to and from TOML and YAML config files.
- [nonideality](nonideality.md) — the reusable analog non-ideality kernels and their config dataclasses.
- [physical_constant](physical_constant.md) — shared SI physical constants and derived quantities.
- [profiler](profiler.md) — the side-channel profiler collecting per-instance PPA events.
- [quant](quant.md) — the stochastic-rounding quantization kernels: floor division, float-to-int scale quantizer, and code-edge bucketize.
- [mixin/](mixin/README.md) — the reusable mixin classes: fabrication, profiling, registry dispatch, and validation.
