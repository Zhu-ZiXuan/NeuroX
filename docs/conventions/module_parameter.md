# Module parameter

A module's parameters are classified along two orthogonal axes: **Source** (where the number comes from) and **config layering** (when it is fixed and who owns it).

## Source

Every parameter in a Reference §Parameters table carries a Source drawn from this fixed taxonomy, which tells a reviewer how far to trust it.

| Source | Definition | Examples |
|---|---|---|
| **Constant** | Universal physical constant; fixed by SI definition, no uncertainty. | $k_B$, $q$ |
| **Measured** | Measured on a real device / silicon; carries a statistical distribution. | RRAM I-V, conductance variation, noise spectra |
| **Process** | Given by the foundry process / PDK / datasheet (nominal). | NMOS per-width capacitance density, process corners |
| **Extracted** | Obtained by parasitic extraction from layout geometry. | interconnect segment $R/C$ |
| **Design** | Freely chosen by the chip / architecture designer within constraints. | row/col count, ADC bits, RRAM state count, WL pulse width, digit radix, NMOS width |
| **Calibrated** | Fitted by a NeuroX calibration procedure. | ADC `rescale_factor` (vs physical data), solver iteration counts (vs numerical convergence) |

Rules:

- **Calibrated** — state in the parameter's row what it is calibrated against: *physical data* (e.g. ADC `rescale_factor`) or *numerical convergence* (e.g. solver iteration counts, which are chip-independent). Numerical-convergence parameters are numerical control parameters, not physical-data fits: they are tuned for solver convergence and so affect speed and stability, not physical fidelity.
- **Derived quantities** have no separate source. Label a derived parameter with its dominant upstream source and note the derivation in the row — e.g. thermal voltage $V_T = k_B T / q$ → Constant-derived, evaluated at runtime $T$ (the constants $k_B$, $q$ are exact, but the value is evaluated at the operating temperature and so inherits the $T$ variation).
- **Distinguish near neighbours**: Measured (we/partners measured it) vs Process (foundry nominal); Process (process density) vs Extracted (layout-dependent geometry); Design (freely chosen, e.g. NMOS width) vs Process (locked, e.g. per-width cap density).
- **Runtime inputs** (activations, weights, temperature $T$, ADC operating point) are inputs, not parameters — never list them in §Parameters.

## Config layering

Every `config` parameter belongs to one of four layers, distinguished by when it is fixed and who owns it:

- **Process** — fixed by process selection; changes only when the process changes.
- **Design** — freely chosen within one process: transistor sizing (`W__um`, `L__um`), capacitor sizing, reference voltages, op-amp target gain.
- **Spec** — performance, statistics, and bookkeeping: noise, mismatch, area, leakage, latency.
- **Runtime** — known only at run time: shape, active ADC mode, active bit width, operating temperature when treated as an environment input, and per-call tensors.

A design parameter binds to the holder's design rather than the module's intrinsic nature, so it is a field on the holder's config and a keyword `__init__` argument, not on the module's own config. Runtime parameters never appear on a config dataclass. A module's own doc declares each parameter's layer, and thereby which of its parameters are design parameters.
