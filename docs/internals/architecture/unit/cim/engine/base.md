# CimEngine

`CimEngine` composes four independently configurable stages around one
`CimMacro`: geometric placement, selected-input activation, weight-slice
layout, and input-slice serialization. The engine itself owns lifecycle
orchestration, the macro child, value-range delegation, and execution order.

## Design decisions

- **One engine, composed stages.** Direct, inter-plane, and intra-port weight
  layouts are not engine subclasses. `CimEngineConfig` contains
  `placement`, `input_activation`, `weight_slice`, and `x_slice`; only the
  latter two use registry dispatch because they have alternative
  implementations.
- **Mapping and aggregation remain paired.** A stage that introduces an axis
  also owns the digital module that removes it. This prevents a mapping
  strategy from being combined with an incompatible aggregation path and
  keeps the digital PPA model attached to the operation it represents.
- **The macro boundary remains logical.** The engine supplies `input_num` and
  `output_num` when constructing the macro and reads only its public value
  ranges, activation limit, and ADC metadata. It does not inspect rows,
  columns, cell digits, or readout topology.
- **Canonical macro instance layout.** Every configuration uses
  `[M=1, Sx=1, Sw, Tc, G]`. Direct and intra-port weight layouts keep
  a structural `Sw=1` axis. Fixed size-one axes make all stage combinations
  follow one execution graph.
- **Containers do not report duplicate PPA.** The engine and all four stages
  set `is_profile_target = False`; their macro and digital children report
  physical PPA. A stage is a mapping construct holding no circuit, so it has
  no area, no leakage and no duration — the three PPA quantities agree on it.
- **The engine times the schedule; the stages only declare its axes.** Every
  serial axis below the unit is engine-inserted: `M` from its caller, `Sx` from
  `x_slice.slice_num`, `D` from `placement.block_step_num`, `P` from
  `input_activation.input_phase_num`. One macro access serves each
  `(M,Sx,D,P)` point, so `latency__ns` multiplies the macro rather than summing
  it, and `Sw`, `Tc` and `G` never multiply anything because they
  are parallel silicon. The digital blocks own no axis of the schedule either,
  so the engine reads each embedded block's `latency_per_op__ns` through its
  stage and supplies the counts itself — it introduced those axes, so it is
  the layer that multiplies them: the ports one operation covers
  (`output_num` for the two accumulators, `weight_slice.aggregated_output_num`
  for the two reconstructions) and how often that operation happens. The phase accumulator folds successive
  arrivals into one register, so it runs once per macro access; the adder-tree
  and positional-sum reductions close their whole axis in one window, so they
  run once per step that axis completes on.
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
`[..., M=1, Sx=1, Sw, Tc, G, input_num, output_num]`.

## Matmul path

`matmul(input)` executes:

1. `x_slice.slice`: append logical `Sx`.
2. `placement.organize_x`: form `[M,Sx,Sw=1,Tc,G=1,L]`.
3. `input_activation.unroll_input_phases`: insert `P` and mask each local
   input group.
4. `placement.unroll_block_steps`: insert `D` and route each local block into
   its geometric slot.
5. `cim_macro.vec_mat_mul`: produce one code per macro read and convert it to
   `int64` at the analog-to-digital boundary.
6. Aggregate in the fixed order `P -> Tc -> Sw -> Sx`.
7. `placement.restore_output`: reorder `(D,G,Q)`, flatten, and trim to `N`.

`Tc` is reduced before either precision axis. Consequently the contraction
accumulator's instance multiplicity includes the physical `Sw` macro planes
for inter-plane layouts, matching the tensor on which it operates.

## Contracts

- A mapping stage must preserve all axes it does not own.
- `D` identifies different logical output blocks and is never reduced.
- `P` and `Tc` are ordinary sums; `Sw` and `Sx` are radix-weighted sums.
- All absent axes remain explicit with extent one.
- The engine converts macro output codes to `int64`; digital modules do not
  perform dtype conversion.
- `ideal_macro=True` constructs the configured macro first and then calls
  `to_ideal()` without changing placement or aggregation.

---

- **Reference**: [engine family](../../../../../reference/architecture/unit/cim/engine/family.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/base.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_engine_input_packing.py`, `tests/architecture/unit/test_engine_latency.py`
