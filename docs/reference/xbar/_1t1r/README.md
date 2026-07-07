# 1T1R array unit

The concrete 1T1R array type: the cell and the core array built for it. The cell is the two-terminal RRAM-plus-access-NMOS element the solver sees; the core is the pure physical array (cell array, wire parasitics, generic solver) that provides the array steady-state and the array-internal energy.

- [cell](cell.md) — the 1T1R cell (`XbarCell1T1R`): physical model, access-node condensation, signed-conductance contract, device-capacitor energy.
- [core](core.md) — the 1T1R core array (`Core1T1R`): array model, governing equations, energy.
