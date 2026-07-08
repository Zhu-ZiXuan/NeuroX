# Module parameter

A module's parameters are classified along two orthogonal axes: **Source** (where the number comes from) and **config layering** (when it is fixed and who owns it).

## Source

Every parameter in a Reference §Parameters table carries a Source drawn from this fixed taxonomy, which tells a reviewer how far to trust it.

| Source | Definition |
|---|---|
| **Constant** | Universal physical constant; fixed by SI definition, no uncertainty. |
| **Measured** | Measured on a real device / silicon; carries a statistical distribution. |
| **Process** | Given by the foundry process / PDK / datasheet (nominal). |
| **Extracted** | Obtained by parasitic extraction from layout geometry. |
| **Design** | Freely chosen by the chip / architecture designer within constraints. |
| **Calibrated** | Fitted by a NeuroX calibration procedure. |

Rules:

- **Calibrated** — state in the parameter's row what it is calibrated against: *physical data* or *numerical convergence* (chip-independent). Numerical-convergence parameters are numerical control parameters, not physical-data fits: they are tuned for convergence and so affect speed and stability, not physical fidelity.
- **Derived quantities** have no separate source. Label a derived parameter with its dominant upstream source and note the derivation in the row — e.g. thermal voltage $V_T = k_B T / q$ → Constant-derived, evaluated at runtime $T$ (the constants $k_B$, $q$ are exact, but the value is evaluated at the operating temperature and so inherits the $T$ variation).
- **Distinguish near neighbours**: Measured (we/partners measured it) vs Process (foundry nominal); Process (layout-independent nominal) vs Extracted (layout-dependent geometry); Design (freely chosen) vs Process (locked).
- **Runtime inputs** (activations, weights, temperature $T$) are inputs, not parameters — never list them in §Parameters.

## Constraint

A Reference §Parameters table carries a Constraint column giving each parameter's physical valid domain — the range physics allows, such as a positive ratio or a bound like n > 1. Physical validity is model content, so it lives in Reference.

- Write `—` when the parameter is physically unconstrained.
- Write `TODO (domain author)` for a non-obvious bound; never fabricate one.
- State the physical domain only. What the code raises when a value falls outside the domain is runtime-validation behavior and belongs to Internals, not this column.

## Config layering

Every `config` parameter belongs to one of four layers, distinguished by when it is fixed and who owns it:

- **Process** — fixed by process selection; changes only when the process changes.
- **Design** — freely chosen within one process.
- **Spec** — performance, statistics, and bookkeeping.
- **Runtime** — known only at run time.

A design parameter binds to the holder's design rather than the module's intrinsic nature, so it is a field on the holder's config and a keyword `__init__` argument, not on the module's own config. Runtime parameters never appear on a config dataclass. A module's own doc declares each parameter's layer, and thereby which of its parameters are design parameters.
