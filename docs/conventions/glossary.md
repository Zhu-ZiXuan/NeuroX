# Glossary

## Hardware structure

- **cell** — a programmable circuit at an array site, including its devices and terminal relations.
- **array** — the cell grid and its interconnect.
- **macro** — an array together with drive, readout, conversion, and control circuits providing a logical vector-matrix operation.
- **peripheral** — a macro circuit outside the array grid.
- **crossbar** — a topology qualifier, abbreviated `xbar` in code.
- **WL phase** — one word-line drive phase within a physical access; several phases may share a held BL/SL boundary.

## Operations and data

- **fabricate** — establish a static manufacturing realization.
- **program** — replace programmable state using the representation defined by the interface.
- **snapshot / snap** — capture / captured physical state for an access, including any enabled access-level sampling. Numerical iterations reuse the held realization.
- **DC operating point / dcop** — steady-state circuit quantities at specified boundaries.
- **record** — one submitted observation.
- **report entry** — named quantities assembled from completed observations.
- **state** — numerical values and convergence-control data carried between iterations; a terminal state can remain unconverged.
- **trace** — observations from numerical evaluations or their iteration history.

## Signal, code, and value

- **signal** — a physical analog quantity, such as voltage, current, or charge.
- **code** — an integer representation carried by a converter or digital circuit.
- **value** — the numerical quantity represented by signals or codes, such as a logical weight or activation.

## Value domain and slicing

- **value domain** — the range of integer values a particular interface accepts.
- **digit** — one position in a radix encoding. Its mapping to physical devices depends on the circuit topology.
- **slice** — a positional part of a logical value that fits the macro's carrier domain. A slice can contain several native digits.
- **encoding** — an integer's representation by digits and positional weights.
- **coding scheme** — the mapping from logical values to physical storage and readout combinations.

[Precision slicing](../reference/architecture/unit/cim.md#precision-slicing) defines decomposition and reconstruction; [notation](notation_conventions.md) defines the corresponding symbols.
