# device — Implementation

How the device layer is built. The spec is in [reference/device](../../reference/device/README.md); this side covers only what the code cannot tell you — the config / policy / design-parameter split, the fabrication-and-snap contract, and the per-device gotchas.

- [rram](rram.md) — `RRAM` config / init / policy split, programming vs read state, snap contract.
- [nmos](nmos.md) — `NMOS` pure-electrical primitive, precomputed nominals, fabricate-time mismatch.
- [selector](selector.md) — `Selector` threshold-map sampling and broadcast contract.
