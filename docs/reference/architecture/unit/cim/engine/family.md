# CIM engine

`CimEngine` places one logical integer matrix multiplication on [CIM macros](../../../../primitive/macro/cim/family.md). It is a composition of four mappings, each paired with any digital aggregation that reverses it:

- [placement](placement.md): geometric tiling, short-vector packing, block routing, and `Tc` accumulation;
- [input activation](input_activation.md): `max_active_num` grouping and `P` accumulation;
- [weight slicing](weight_slice.md): direct, inter-plane, or intra-port `Sw` layout and weight shift-add;
- [input slicing](x_slice.md): direct or serial `Sx` execution and input shift-add.

These choices are nested stage configurations, not separate engine classes.

## Governing laws

For logical weight `W[N,K]` and input `X[M,K]`, the engine reconstructs

$$Y_{m,n}=\sum_{k=0}^{K-1}X_{m,k}W_{n,k}.$$

Each constituent macro read may include its own analog behavior and ADC quantization; all engine-side mappings and digital reductions are integer operations.

Let `I=input_num`, `Q` be the logical output width selected by the weight layout, and `A=max_active_num`. Placement derives

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

Logical output block `b=dG+g` occupies input slot `[dL,(d+1)L)` of macro group `g`. Missing final blocks and unused input/output positions are programmed to zero but remain in the uniform schedule.

## Execution order

The macro-aligned leading block uses canonical order `[M,Sx,Sw,Tc,G]`; the macro state keeps singleton `M` and `Sx` slots, while the input supplies their runtime extents through ordinary broadcasting. `D` and `P` precede this block as runtime schedule axes. After each macro read, the engine aggregates

$$P\rightarrow T_c\rightarrow S_w\rightarrow S_x,$$

then reorders `(D,G,Q)`, flattens it in logical block order, and trims to `N`. `D` is not a partial-sum axis and is never reduced.

## Configuration

| Field | Meaning |
|---|---|
| `input_num` | logical input ports of one macro |
| `output_num` | logical output ports of one macro |
| `cim_macro_config` | owned macro configuration |
| `placement` | geometric placement and `Tc` accumulator configuration |
| `input_activation` | selected-input grouping and `P` accumulator configuration |
| `weight_slice` | weight layout and optional `Sw` shift-adder configuration |
| `x_slice` | input serialization and optional `Sx` shift-adder configuration |

The policy has the same five owned-child fields. The engine's public `w_value_range` and `x_value_range` come from the two slice stages; ADC metadata and `max_active_num` delegate to the constructed macro. The file-level form of these tables is specified in [configuration](../../../../../api/configuration.md).

## Symbols

| Symbol | Meaning | Code |
|---|---|---|
| $M$ | activation-matrix row count | `input` row dim |
| $S_w,S_x$ | weight and input slice counts | stage configuration |
| $I$ | macro logical input capacity | `input_num` |
| $A$ | maximum selected inputs per read | `cim_macro.max_active_num` |
| $L,Q$ | logical block input/output widths | `placement.plan` |
| $T_c$ | contraction-tile count | `placement.plan.contraction_partition_num` |
| $B,C$ | output-block count and per-macro capacity | `placement.plan` |
| $G,D$ | macro groups and block steps | `placement.plan.block_group_num`, `placement.plan.block_slot_num` |
| $P$ | selected-input phases | `input_activation._input_phase_num` |

$S_x$ is a unit-layer symbol; the macro and solver layers below keep generic leading dims.

## Validation

Bit-exact stage combinations, non-divisible dimensions, coprime dimensions, short-vector packing, and input phases are covered by `tests/architecture/unit/test_cim_unit.py` and `tests/architecture/unit/test_engine_input_packing.py`.
