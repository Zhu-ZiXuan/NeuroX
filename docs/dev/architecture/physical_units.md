# Physical Units

This document records the project-wide convention for physical units, name suffixes, and dtype on the runtime path.

## Tensor units (mandatory)

Every tensor that carries a physical quantity uses one of the following units. The set is closed, self-consistent, and chosen to keep numbers inside the well-conditioned range of `bfloat16` / `float32`:

| Quantity | Unit | Suffix |
|---|---|---|
| Voltage | volt | `__V` |
| Current | microampere | `__uA` |
| Conductance | microsiemens | `__uS` |
| Resistance | megaohm | `__MOhm` |
| Time | nanosecond | `__ns` |
| Capacitance | femtofarad | `__fF` |
| Energy | femtojoule | `__fJ` |
| Power | microwatt | `__uW` |
| Length | micrometre | `__um` |

The set is closed under the products that appear in circuit math:

- `uA · V = uW` (current · voltage = power)
- `uW · ns = fJ` (power · time = energy)
- `fF · V² = fJ` (capacitance · voltage² = energy)
- `MOhm · uA = V` (resistance · current = voltage)
- `uS · V = uA` (conductance · voltage = current)

Tensors **must** be constructed in these units. **No unit conversion is allowed on any tensor-computation path.**

## Config units

Config fields are the **human-interaction surface**. They follow established industrial conventions, even when those differ from the tensor units above (e.g. PDK mobility in `cm² / V / s`, SI physical constants in `J / K` or `C`).

Each config field carries its unit suffix in the field name. No suffix means the field is dimensionless.

Unit conversion from config units to tensor units happens **exclusively inside the consuming class's `__init__`**:

- The frozen config provides raw data and performs `validate*` checks; it never derives or returns processed values.
- The consuming circuit / device class reads `config.foo__some_unit` in its `__init__`, multiplies the conversion factor, stores the result as an internal attribute or buffer in the tensor-unit system.
- Once the runtime path starts, every tensor is already in the canonical units and no further conversion happens.

This rule keeps the runtime path branch-free and unit-pure.

## Suffix grammar

Variable, attribute, buffer, parameter, and field names that carry a physical quantity use the `<name>__<unit>` form:

- Separator: double underscore `__`.
- Unit string: case follows the physical-quantity standard (`V` uppercase for volt, `m` lowercase for metre; SI scale prefixes `T G M k m u n p f` use their standard case).
- Multiplication is implicit — two adjacent unit symbols denote a product. Example: `A_vt__mV_um` = mV · μm.
- Division uses the explicit `_per_` token. Example: `mu0__cm2_per_V_s` = cm² / V / s.
- Dimensionless quantities use no suffix.

The base physics-unit standards themselves (case rules for V/A/m, scale-prefix conventions, etc.) are not re-documented here; they follow accepted physics convention.

## Examples

| Name | Reading |
|---|---|
| `v_dd__V` | supply voltage in volts |
| `T_ref__K` | reference temperature in kelvin |
| `c_unit__fF` | unit capacitance in femtofarads |
| `A_vt__mV_um` | Pelgrom V_th coefficient in mV·μm |
| `mu0__cm2_per_V_s` | low-field mobility in cm²/V/s |
| `K_BOLTZMANN__J_per_K` | Boltzmann constant in J/K |

## Dtype on the runtime path

| Domain | Dtype | Notes |
|---|---|---|
| Digital computation | `int32` | Standard MAC / shift / requantize result. |
| Digital table indexing | `int64` | Only for table lookups. |
| Analog forward path | dtype chosen at construction; calibrated per build | `bfloat16` may be insufficient when low-LSB noise must remain visible. The calibration tool determines the appropriate analog dtype; treat it as fixed once chosen. |
| Solver internals | dtype hard-coded per solver; calibrated | Likewise determined by the calibration tool; the solver does not switch dtype at runtime. |
| Output rescale (multiplier / rshift) | `float32` | Data fits inside `int16`. `int32` is only used for GPU friendliness, not range. |

Dtype must not change between calls of the same `@torch.compile`-decorated function — a dtype change forces a recompile.
