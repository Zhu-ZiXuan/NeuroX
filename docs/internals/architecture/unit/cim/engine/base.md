# CimEngine

`CimEngine` composes three independently configurable stages around one
`CimMacro`: geometric placement, weight-slice layout, and input-slice
serialization. The engine itself owns only lifecycle orchestration, the macro
child, value-range delegation, and the fixed execution order.

## Design decisions

- **One engine, composed stages.** Direct, inter-plane, and intra-port weight
  layouts are not engine subclasses. `CimEngineConfig` contains
  `placement`, `weight_slice`, and `x_slice`; only the latter two use registry
  dispatch because they have alternative implementations.
- **Mapping and aggregation remain paired.** A stage that introduces an axis
  also owns the digital module that removes it. This prevents a mapping
  strategy from being combined with an incompatible aggregation path and
  keeps the digital PPA model attached to the operation it represents.
- **The macro boundary remains logical.** The engine supplies `input_num` and
  `output_num` when constructing the macro and reads only its public value
  ranges, activation limit, and ADC metadata. It does not inspect rows,
  columns, cell digits, or readout topology.
- **Canonical macro instance layout.** Every configuration uses
  `[*w_batch, M=1, Sa=1, Sw, Tc, G]`. Direct and intra-port weight layouts keep
  a structural `Sw=1` axis. Fixed size-one axes make all stage combinations
  follow one execution graph.
- **Containers do not report duplicate PPA.** The engine and all three stages
  set `is_profile_target = False`; their macro and digital children report
  physical PPA.
- **No engine registry.** `CimEngine.from_config` constructs `CimEngine`
  directly. Polymorphism is limited to the two stage roots where behavior
  actually varies.

## Program path

`program(weight)` checks the shape bound at construction and executes:

1. `weight_slice.slice`: append logical `Sw`.
2. `placement.partition_weight`: form `[D,G,Q,Tc,L,Sw]`.
3. `weight_slice.arrange_weight`: map `Sw` to macro planes or output ports.
4. `placement.pack_weight`: merge the `D` slots into the macro input axis.
5. `cim_macro.program`: store the final macro-native tensor.

The resulting tensor always has
`[..., M=1, Sa=1, Sw, Tc, G, input_num, output_num]`.

## Matmul path

`matmul(input)` executes:

1. `x_slice.slice`: append logical `Sa`.
2. `placement.organize_x`: form `[M,Sa,Sw=1,Tc,G=1,L]`.
3. `placement.unroll_input_schedule`: insert serial `[D,P]`.
4. `cim_macro.vec_mat_mul`: produce one code per macro read and convert it to
   `int64` at the analog-to-digital boundary.
5. Aggregate in the fixed order `P -> Tc -> Sw -> Sa`.
6. `placement.restore_output`: reorder `(D,G,Q)`, flatten, and trim to `N`.

`Tc` is reduced before either precision axis. Consequently the contraction
accumulator's instance multiplicity includes the physical `Sw` macro planes
for inter-plane layouts, matching the tensor on which it operates.

## Contracts

- A mapping stage must preserve all axes it does not own.
- `D` identifies different logical output blocks and is never reduced.
- `P` and `Tc` are ordinary sums; `Sw` and `Sa` are radix-weighted sums.
- All absent axes remain explicit with extent one.
- The engine converts macro output codes to `int64`; digital modules do not
  perform dtype conversion.
- `ideal_macro=True` constructs the configured macro first and then calls
  `to_ideal()` without changing placement or aggregation.

---

- **Reference**: [engine family](../../../../../reference/architecture/unit/cim/engine/family.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/base.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_engine_input_packing.py`
