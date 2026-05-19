# `neurox/device/selector.py`

## Current role

`Selector` models an OTS (Ovonic Threshold Switch) selector device with a per-cell threshold-voltage map. It is a device-layer primitive; consuming circuits decide how to use the threshold map.

## Config boundary

`SelectorConfig` carries:

- `vth_nominal__V` — nominal threshold voltage [V].
- `vth_mismatch__V` + `enable_vth_mismatch` — additive Gaussian mismatch sigma on `vth__V` and its toggle.

Device-level only. Array shape is **not** a config field — it is supplied as an explicit `__init__` argument because shape is a property of the deployment instance, not the device itself.

## Construction

`Selector.__init__(*, cfg, T__K, dtype, array_shape)`:

- `cfg` — process / spec.
- `T__K`, `dtype` — uniform device-construction signature (currently unused by the selector model but stored for future use).
- `array_shape` — the spatial shape over which the static threshold map is fabricated.

Devices do not own a `name` — selector PPA rolls up to the consuming circuit.

## Sampling behaviour

`sample_vth_like(reference)` returns a threshold-voltage tensor broadcast against the reference tensor:

- in `training` mode, a fresh Gaussian draw is taken on each call
- in `eval` mode, a single static draw is cached in `_vth_static__V` and reused for all subsequent calls

This split mirrors a real chip: training sees the full mismatch distribution; inference uses one fixed fabricated array.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/architecture/state_holding.md`
