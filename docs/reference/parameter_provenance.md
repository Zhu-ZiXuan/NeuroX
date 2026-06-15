# Parameter Provenance

Every parameter in a Reference §Parameters table carries a **Source** drawn from this fixed taxonomy. Source answers "where does this number come from", which is what tells a reviewer how far to trust it. This is orthogonal to the code-side config layering (`process` / `design` / `spec` / `runtime`, which answers "when is it fixed and who owns it") — that layering is documented with the configuration schema, not here.

| Source | Definition | Examples |
|---|---|---|
| **Constant** | Universal physical constant; fixed by physics, no uncertainty. | $k_B$, $q$ |
| **Measured** | Measured on a real device / silicon; carries a statistical distribution. | RRAM I-V, conductance variation, noise spectra |
| **Process** | Given by the foundry process / PDK / datasheet (nominal). | NMOS per-width capacitance density, process corners |
| **Extracted** | Obtained by parasitic extraction from layout geometry. | interconnect segment $R/C$ |
| **Design** | Freely chosen by the chip / architecture designer within constraints. | row/col count, ADC bits, RRAM state count, WL pulse width, digit radix, NMOS width |
| **Calibrated** | Fitted by a NeuroX calibration procedure. | ADC `rescale_factor`, solver iteration counts |

Rules:

- **Calibrated** — state in the parameter's row what it is calibrated against: *physical data* (e.g. ADC `rescale_factor`) or *numerical convergence* (e.g. solver iteration counts, which are chip-independent). Numerical-convergence parameters affect speed and stability, not physical fidelity.
- **Derived quantities** have no separate source. Label a derived parameter with its dominant upstream source and note the derivation in the row — e.g. thermal voltage $V_T = k_B T / q$ → Constant (derived from $k_B$, $q$, $T$).
- **Distinguish near neighbours**: Measured (we/partners measured it) vs Process (foundry nominal); Process (process density) vs Extracted (layout-dependent geometry); Design (freely chosen, e.g. NMOS width) vs Process (locked, e.g. per-width cap density).
- **Runtime inputs** (activations, weights, temperature $T$, ADC operating point) are inputs, not parameters — never list them in §Parameters.
