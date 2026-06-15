# `neurox/device/selector.py`

## Current role

`Selector` models an OTS (Ovonic Threshold Switch) selector device with a per-cell threshold-voltage map. It is a device-layer primitive; consuming circuits decide how to use the threshold map.

## Config boundary

`SelectorConfig` carries:

- `vth_nominal__V` — nominal threshold voltage [V].
- `vth_mismatch__V` — additive Gaussian mismatch sigma on `vth__V` (gated by `SelectorPolicy.vth_mismatch`).

Device-level only. The per-instance shape is **not** a config field — it is supplied as an explicit `__init__` argument because shape is a property of the deployment instance, not the device itself.

## Policy boundary

`SelectorPolicy` carries the per-run decision:

- `vth_mismatch` — apply `vth_mismatch__V` per cell at fabricate time.

## Construction

`Selector.__init__(*, config, policy, inst_shape, dtype, T__K)`:

- `config` — process / spec (`SelectorConfig`).
- `policy` — runtime mismatch switch (`SelectorPolicy`).
- `inst_shape` — per-instance fabrication shape over which the static threshold map is sampled.
- `dtype`, `T__K` — uniform device-construction context (`T__K` currently unused by the selector model but stored for future use).

Devices do not own a `name` — selector PPA rolls up to the consuming circuit.

## Sampling behaviour

`Selector` inherits `FabricateMixin`. Each `fabricate()` call resamples the static V_th map via `_sample_fabricate_mismatch()`, populating `vth__V` at `(*self._inst_shape,)` from `nominal_vth__V`.

`sample_vth_like(reference)` returns `vth__V` broadcast to the reference tensor's shape, device, and dtype. Resampling cadence is owned by the caller's training / inference loop, which calls `model.fabricate()` per training step (for noise-aware training) or once at inference setup.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/architecture/state_holding.md`
- `docs/dev/architecture/fabrication_lifecycle.md`
