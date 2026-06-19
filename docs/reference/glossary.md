# Glossary

One- or two-sentence definitions of the terms used across the [Reference](README.md) manual, for algorithm, architecture, and device readers. Symbols and units are pinned separately in [notation_conventions](notation_conventions.md); parameter Source terms are in [parameter_provenance](parameter_provenance.md). A term links to its document only where that document exists; otherwise the code name is set in `code` font.

## Hardware structure

- **xbar (crossbar)** — the physical tile that performs one compute-in-memory VMM: cells whose programmed conductances multiply an input vector and accumulate as analog current, which a readout digitizes. The abstract contract is in [xbar/base](xbar/base.md); the family overview is [xbar](xbar/README.md).
- **tile** — synonym for one xbar instance: the unit that carries the primitive `program` and `VMM read` operations and publishes its integer value domain. The macro/xbar interface decides the value range above the tile; the tile publishes only what it can physically carry.
- **macro** — the simulated hardware architecture that organizes crossbar tiles along two orthogonal axes: matrix tiling ($T_r$, $T_c$) of a weight across xbars, and precision slicing ($S_w$, $S_a$) that decomposes each value into slices. It aggregates the per-tile, per-slice partial products by a radix-weighted shift-add into the operator-facing integer matmul. See [macro](macro/README.md).
- **circuit_core** — the shared physical 1T1R cell array: it encapsulates and solves the array only (cells, interconnect ladder, boundary clamp-drivers) and carries no encoding or readout. Operating xbars build on it by adding a readout. See [circuit_core](xbar/_1t1r/circuit_core.md).
- **operating xbar** — a concrete xbar that composes a `circuit_core` with a readout chain and a coding scheme to turn the physical array into a signed VMM. The current member is the offset-coded xbar; a differential xbar is reserved.
- **offset (offset-coded) xbar** — the operating xbar whose read is differential, data minus a per-group reference column, so the reference cancels a common-mode offset and the signed result carries the sign of the dot product. See [offset](xbar/_1t1r/offset.md).
- **differential xbar** — a reserved sibling operating xbar that reuses the same `circuit_core` with a different coding and readout; not yet implemented.
- **clamp-driver** — a boundary actor that holds its array node at a target clamp voltage and absorbs whatever port current the array draws, supplying one of the two boundary constraints of a column. The SL [driver](analog/driver.md) is the SL clamp-driver (an ideal, zero-output-impedance source holding the SL node) and the [TIA](analog/tia/README.md) is the BL clamp-driver (a finite-gain virtual-ground hold on the BL node). See [driver](analog/driver.md) and [tia](analog/tia/README.md).
- **readout** — the voltage-domain chain from an array's per-column boundary output to signed ADC codes (switch-cap, mux, differential ADC). An operating xbar composes a readout with its core. See [readout](xbar/readout/README.md).
- **topology family** — a grouping of xbars sharing one cell wiring scheme, named by devices per cell: **1T1R** (one access transistor, one RRAM; the only implemented family), with **2T1R** and **2T2R** reserved. See [xbar/_1t1r](xbar/_1t1r/README.md).

## Operations and lifecycle

- **VMM (vector-matrix multiply)** — the primitive a tile computes: drive an input activation onto the rows, settle to the DC operating point, and digitize the per-column accumulated result to a signed integer code.
- **DCOP (DC operating point)** — the steady-state node-voltage and branch-current solution of the array under the interconnect parasitics and boundary clamp-drivers, found by the [solver](xbar/_1t1r/solver.md) via damped Newton iteration. The `circuit_core` holds no DCOP of its own; the per-call solve state is a transient container.
- **program** — write a weight into the tile: a non-negative integer digit tensor whose digits combine positionally with radix $r$. Writes the actual weight state; may be called any number of times, each overwriting the previous encoding.
- **fabricate** — resample static manufacturing variation across every owned module. Re-callable; each call re-samples mismatch from the unchanged nominal template, with no state accumulating across calls. Orthogonal to `program`.
- **snapshot** — materialize per-call dynamic noise into a transient per-call container of dynamic noise that the solve / convert / VMM consumes; it is never persisted on the module. Distinct from the persistent state stages (nominal, actual) where the snapshot stage is this transient one.

## Value domain and slicing

- **value domain** — the integer grid a tile can physically carry: the input grid $\mathcal{X}$, the digit count $D$, and the radix $r$. The tile publishes its value domain; algorithm-side ranges map onto it. The algorithm-side `value_range` (complete value an input scalar can take) is kept distinct from the primitive single-cell `digit_range`. The per-slice value range and the **slice_radix** $R$ are published by the xbar interface (the authority), not inferred from above.
- **digit** — the level-0 value-domain unit: the integer symbol one xbar cell carries, in radix **digit_radix** $r$, bounded by the per-cell `digit_range`. The physical analog quantity it maps to is the conductance state/level (see device). The count of digits per slice is `digit_count` ($D$).
- **slice** — the level-1 unit (synonyms weight-slice, one-digit slice): a fixed-capacity positional piece of a value made of `digit_count` digits at radix $r$, so its positional ratio is the **slice_radix** $R = r^{\,\mathrm{digit\_count}}$. Its per-slice value range is computed from `digit_count` and the radix and is published by the xbar interface (the authority). The slice count ($S_w$ on the weight side, $S_a$ on the input side) is config-given, not inferred. Slices recombine into the value by the radix-weighted shift-add, the inverse of decomposition. See [macro/xbar](macro/xbar/README.md) and [digital/shift_adder](digital/shift_adder.md).
- **value** — the level-2, role-neutral algorithm scalar: a weight on the weight side, an activation on the input side. It is decomposed into slices for the macro and recombined by the radix-weighted shift-add. The xbar carries values role-neutrally and asserts no weight/activation application role.
- **digit_radix** ($r$) — the per-digit radix: the base in which one cell's digit is expressed. It enters the radix fold and is published by the xbar interface (`w_digit_radix`).
- **slice_radix** ($R$) — the positional weight ratio between adjacent slices, $R = r^{\,\mathrm{digit\_count}}$, derived from the xbar `digit_count` and `digit_radix`. The LSB-first slice weights $(1, R, R^2, \ldots)$ drive the shift-add recombination $M = \sum_i m_i R^i$.
- **$T_r$ / $T_c$** — the general matrix-tiling counts: $T_r = \lceil N / N_{\mathrm{col}} \rceil$ is the output-axis tile count (rows of the transposed weight) and $T_c = \lceil K / N_{\mathrm{row}} \rceil$ is the contraction-axis tile count. This matrix-tiling axis applies to any matmul and is distinct from the CIM-specific precision-slicing axis $S_w$, $S_a$.
- **encoding (codec)** — the generic integer-to-digit-string codec over (radix, `digit_count`, policy), selected by the string discriminator `"true_form" | "complement" | "canonical"`. It is an implementation primitive with no slicer or application semantics.
- **offset coding** — the coding scheme of the offset xbar: a logical column maps to physical columns in data-major / digit-minor order with a reference column inserted per group; the logical-to-physical scatter is applied at program time, the data-minus-reference subtraction at readout. See [offset](xbar/_1t1r/offset.md).

## Output and references

- **rescale factor** ($s$) — the scalar mapping a read's signed integer code back to the ideal integer dot product, with $M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s$. A physical tile reads $s$ from a calibrated `(mode, b)` table; the ideal twin derives it from integer geometry alone. See [xbar/base §Output rescale](xbar/base.md#output-rescale).
- **ideal twin** — the lossless tile-level reference for any physical xbar: it preserves the primitive operation, the value domain, and the output integer grid but discards every analog non-ideality (IR drop, noise, finite gain). It is the integer truth a physical read is compared against, and ADC calibration uses the physical-vs-ideal gap. See [xbar/base §Ideal twin](xbar/base.md#ideal-twin).
- **PPA (power, performance, area)** — the cost-model accounting reported per run: static area and leakage from each circuit's config, plus dynamic energy and latency events emitted during forward. The cost model lives in Reference (e.g. the [circuit_core energy model](xbar/_1t1r/circuit_core.md#energy-model)); the profiler mechanism that collects it is an implementation concern.

---

- **Notation**: [notation_conventions](notation_conventions.md)
- **Parameter provenance**: [parameter_provenance](parameter_provenance.md)
- **Manual root**: [reference](README.md)
