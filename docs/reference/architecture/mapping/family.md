# CIM operator mapping

CIM units place logical integer matrix multiplications on [CIM macros](../../primitive/macro/cim/family.md). Each operator composes mappings, each paired with any digital aggregation that reverses it:

- [tiling](placement.md): capacity-bounded logical blocks and contraction accumulation;
- optional input-slot sharing: physical block grouping and input routing;
- [input activation](input_activation.md): `max_active_num` grouping and `P` accumulation;
- [weight slicing](weight_slice.md): direct, inter-plane, or intra-port `Sw` layout and weight shift-add;
- [input slicing](x_slice.md): direct or serial `Sx` execution and input shift-add.

Tiling and merging are geometric transformations. Each operator owns its phase accumulator and configured precision reconstruction circuits.

## Governing laws

For logical weight `W[N,K]` and input `X[M,K]`, the mapped operator reconstructs

$$Y_{m,n}=\sum_{k=0}^{K-1}X_{m,k}W_{n,k}.$$

Each constituent macro read may include its own analog behavior and ADC quantization; all unit-side mappings and digital reductions are integer operations.

Let `I=input_num`, `Q` be the logical output width selected by the weight layout, and `A=max_active_num`. Tiling and optional sharing derive

$$
L=\min(K,I),\quad
T_c=\left\lceil\frac{K}{L}\right\rceil,\quad
B=\left\lceil\frac{N}{Q}\right\rceil,\quad
C=\left\lfloor\frac{I}{L}\right\rfloor,
$$

$$
G=\left\lceil\frac{B}{C}\right\rceil,\quad
D=\left\lceil\frac{B}{G}\right\rceil,\quad
P=\left\lceil\frac{L}{A}\right\rceil.
$$

Linear units place output tiles on separate macros without input-slot sharing. Convolution units may enable sharing; without it, take `C=1`, giving `G=B` and `D=1`.

Logical output block `b=dG+g` occupies input slot `[dL,(d+1)L)` of macro group `g`. Missing final blocks and unused input/output positions are programmed to zero. Each block carries its valid logical output count; the macro determines the corresponding lane/scan activity.

## Slice capacity

The macro's accepted value interval bounds each positional slice independently.
For radix $r$, unsigned digits occupy $[0,r-1]$; true-form and canonical signed
digits occupy $[-(r-1),r-1]$. Complement encoding uses $[0,r-1]$ for every lower
digit and $[-\lfloor r/2\rfloor,\lceil r/2\rceil-1]$ for the highest digit.
The radix is the largest integer at least two for which these digit intervals
fit the macro's accepted range. With one complement slice, only the highest-digit
constraint applies.

A nonnegative carrier requires unsigned decomposition. Signed encodings require
negative digit capacity even when an operation's values happen to be nonnegative.
Inputs outside the representable range may change on reconstruction, while each
emitted digit remains within the carrier's bounds. Numerical reconstruction remains
the same positional sum for both weights and activations.

## Execution order

Precision slicing precedes both geometric transforms. Each precision slice is tiled
independently, then placed in a macro input slot. Input phases select portions of the
placed input without changing its weight placement.

After macro reads, input phases are accumulated. Merging is reversed, then tiling
recovery restores the weight-slice positions and sums input-tile partial products. Weight
and input precision slices are then reconstructed in that order. Output tile recovery
concatenates valid outputs and discards padding; merge steps are never summed.

Geometric recovery has no modeled accumulator energy, area, or latency. The input-phase
accumulator and precision reconstruction circuits retain their own accounting.

## Configuration

| Field | Meaning |
| --- | --- |
| `cim_macro_config` | owned macro configuration, including its fixed input, lane, and scan geometry |
| `merge` | whether convolution blocks share macro input capacity |
| `phase_accumulator_config` | serial accumulator for the input phases |
| `w_slice_num`, `x_slice_num` | positional slice counts |
| `w_slice_encoding`, `x_slice_encoding` | positional encodings, or no decomposition |
| `tiling` | tile strategy placing weight slices on separate planes or within each tile |
| `w_shift_adder_config`, `x_shift_adder_config` | optional precision reconstruction circuits |

Policy fields correspond to the configured children. The unit reads `input_num` and derived `output_num` from the constructed macro; its public `w_value_range` and `x_value_range` come from the two slicers, while ADC metadata and `max_active_num` also delegate to the macro. The file-level form of these tables is specified in [configuration](../../../api/configuration.md).

## Symbols

| Symbol | Meaning | Code |
| --- | --- | --- |
| $M$ | activation-matrix row count | `input` row dim |
| $S_w,S_x$ | weight and input slice counts | slicer configuration |
| $I$ | macro logical input capacity | `input_num` |
| $A$ | maximum selected inputs per read | `cim_macro.max_active_num` |
| $L,Q$ | logical block input/output widths | `tiling.tile_input_num`, `tiling.logical_tile_output_num` |
| $T_c$ | contraction-tile count | `tiling.in_tile_num` |
| $B,C$ | output-block count and per-macro capacity | `tiling.out_tile_num`, `merge.input_slot_capacity` |
| $G,D$ | macro groups and block steps | `merge.macro_group_num`, `merge.merge_step_num` |
| $P$ | selected-input phases | `input_activation.input_phase_num` |

$S_x$ is a unit-layer symbol; the macro and solver layers below keep generic leading dims.

## Validation

Bit-exact stage combinations, non-divisible dimensions, coprime dimensions, short-vector packing, and input phases are covered by `tests/architecture/unit/linear/test_cim_mapping.py` and `tests/architecture/unit/conv2d/test_cim_mapping.py`.
