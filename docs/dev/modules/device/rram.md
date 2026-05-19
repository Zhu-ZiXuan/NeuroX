# `neurox/device/rram.py`

## Current role

`RRAM` owns the programmed and read-time behaviour of one resistive memory device array:

- state-to-conductance mapping
- programming noise
- drift
- read noise and stuck-at effects
- programmed conductance state ownership

## Config boundary

`RRAMConfig` remains a device config:

- process-like device parameters
- device-level non-ideality / spec parameters
- intrinsic top / bottom parasitic capacitances of the memory cell

Unlike the access-NMOS parasitics, these capacitances are treated as intrinsic to the device / cell model rather than circuit-owned layout choices.

## State holding

`RRAM.fabricate(...)` / `program(...)` materializes programmed conductance state inside the `RRAM` instance. Runtime reads consume explicit snapshots rather than forcing parent circuits to duplicate the state.
