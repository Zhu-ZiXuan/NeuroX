# `neurox/analog/decoder.py`

## Current role

`Decoder` is the WL row-address path. It owns the decoder + WL-driver energy / latency accounting and is the circuit that hands integer row codes to the WL DAC.

## Modes

The decoder operates in one of two modes, chosen by `cfg.bit_serial`:

- **Parallel-multibit** (`bit_serial=False`): per-row integer codes are forwarded directly to `dac.convert(codes)`. One DAC call per decode operation.
- **Bit-serial** (`bit_serial=True`): each per-row integer is expanded into `n_address_bits` single-bit codes (LSB-first); a new bit-cycle axis is inserted at `dim=-2`, one bit per cycle. The DAC then sees codes in `{0, 1}` per cycle.

The decoder is a leaf analog circuit, not a polymorphic family — no `from_config` dispatcher.

## Energy / latency

Per-call dynamic energy is:

```
E = n_address_bits · c_gate__fF · v_dd__V²  +  e_overhead__fJ
```

Per-call latency is `n_address_bits · t_gate__ns`. Both are emitted through the profiler side channel.

## Construction

`Decoder.__init__(*, cfg, name, inst_shape, dtype, T__K)`:

- `cfg`, `name`, `inst_shape` — design config, profiler name, and per-instance fabrication shape.
- `dtype`, `T__K` — accepted for the uniform analog construction signature but unused by today's behavioural decoder model.

`Decoder` inherits `FabricateMixin`; it has no static mismatch state, so the inherited `_sample_fabricate_mismatch` is the default no-op.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/modules/analog/dac/README.md`

