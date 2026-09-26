# CIM execution

CIM execution maps a logical operator onto finite hardware, schedules accesses, and assembles results. The operator defines the computation, the unit composes mapping and scheduling, and the macro implements one access.

## Mapping responsibilities

The operator supplies logical input and weight dimensions; the macro supplies capacities, value ranges, and selection limits. The unit composes accesses against those constraints without interpreting the macro's internal cell layout. Precision representation, physical placement, recovery order, and their governing equations are specified by the [CIM unit model](../reference/architecture/unit/cim.md).

The unit owns its arithmetic circuits. Mapping tools reference those circuits without introducing another hardware owner. Lane arithmetic belongs to the digital children; storage for partial sums belongs to the unit-local peripheral budget. An ideal macro provides a control through the same placement and recovery path.

Compiled execution covers operator layout transforms, matrix execution, recovery, bias, and profiling for both CIM and ideal implementations. Every invocation preserves its observation submissions; construction and programming remain explicit lifecycle events.

## Hardware multiplicity and reuse

Replication determines hardware population; temporal reuse determines access count and duration. Changing the number of workload output positions can change work without changing hardware. Input-slot sharing reuses each allocated macro within the mapped operator.

Grouped convolution gives each group an independent macro population and local and global digital circuits. Mapping and input-slot sharing operate within each group; output recovery concatenates groups without summing them. Group multiplicity contributes to hardware cost and energy while group schedules run in parallel.

## Access metadata

Placement and valid output counts derive from the same unpadded geometry, including weight slices, partial tiles, and absent input slots. Different reuse slots can have different valid widths. Zero-valued weights remain valid. The placement owner keeps this metadata aligned with programmed weights, and the macro uses it consistently for electrical execution and timing. Local recovery preserves that correspondence until outputs are restored to logical order.

## Timing and accounting

Timing follows [operation-duration ownership](ppa_accounting.md#operation-duration). The unit combines macro durations with its digital work under the [CIM timing model](../reference/architecture/unit/cim.md#timing), including the scan overlap and global recovery pipeline. Numerical batching preserves this physical schedule.

Circuits account for the operations and hardware they own. The task owner composes the unit's basic operations at retained batch, token, and timestep positions and determines any cross-unit overlap.

## One physical access

A macro establishes electrical boundaries, schedules their phases, and owns peripheral sampling. It distinguishes a boundary held across phases from a fresh boundary event. The array evaluates its cells and interconnect under those supplied conditions and returns the terminal currents and voltages required by surrounding circuitry.

Numerical iterations toward the operating point add no physical phases. Shared boundaries follow [physical-state lifetimes](physical_state.md); their establishment and phase-dependent activity contribute energy independently of numerical batching.
