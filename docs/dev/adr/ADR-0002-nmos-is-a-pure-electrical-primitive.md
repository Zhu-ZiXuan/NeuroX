# ADR-0002: NMOS Is a Pure Electrical Primitive

## Status

Accepted

## Context

Older NMOS code mixed two concerns:

- electrical equation evaluation (`Ids`, `gm`, `gds`, mismatch)
- layout-dependent parasitic capacitance lumping (`c_gs`, `c_gd`, `c_db`, `c_sb`)

The parasitic terms are strongly tied to the consuming circuit's layout and structure. They do not belong to the narrow electrical primitive itself.

## Decision

`NMOS` now owns only:

- electrical process parameters
- mismatch / spec parameters
- explicit design parameters `W__um`, `L__um`
- fabricated `beta__uA_per_V2` / `vth__V`
- electrical solve helpers

Parasitic capacitances are owned by the consuming circuit.

For the 1T1R path:

- `CircuitCore1T1RConfig` owns the access-NMOS sizing
- the same config also owns `c_gs_per_um__fF`, `c_gd_per_um__fF`, `c_db_per_um__fF`
- `CircuitCore1T1R` computes its own lumped parasitic scalars from those per-width densities

## Consequences

Positive:

- `NMOS` is lighter and more reusable
- circuit ownership of parasitics is explicit
- device config no longer carries layout-specific parasitic fields

Tradeoff:

- consuming circuits must carry their own parasitic-cap configuration
- multiple circuits may use different lumping rules for the same underlying transistor model
