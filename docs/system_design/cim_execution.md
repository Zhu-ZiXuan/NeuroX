# CIM execution

A CIM execution maps a logical operator onto finite hardware, schedules accesses, and assembles their results. The operator defines the computation, the engine determines its mapping and schedule, and the macro implements one access.

## Mapping a workload

The workload supplies logical input and weight dimensions. The macro supplies fixed logical capacities, value ranges, and selection limits. Mapping partitions work against those capacities without requiring the engine to interpret the macro's internal rows, columns, or readout circuits.

Digit encoding and slicing represent values in the ranges supported by the hardware. Placement assigns the resulting work to hardware instances and reuse slots. Input activation divides an access when the hardware cannot activate every required input together.

The mapping distinguishes parallel replication from temporal reuse. Replication determines the hardware population; reuse determines the access count and duration. A workload's number of output positions can change the required work without changing the constructed hardware.

## Mapping and aggregation

Each mapping has a corresponding aggregation: digit slices require positional weighting, contraction partitions require summation, and output partitions require restoration to logical order. Their pairing determines both numerical meaning and the digital hardware used to assemble results.

Engine-side transformations use exact integer arithmetic. Padding represents zero, and trimming removes padded positions. The physical macro access introduces the modeled analog and conversion effects. An ideal access therefore provides a control for the same mapping and aggregation.

Input generation and requantization select int32 for quantized inputs and weights. Intermediate interfaces require integer values rather than a particular storage width; each operation states only its own numerical and representation requirements. Lookup operations convert their index expressions locally. ADCs and physical macro accesses produce int32 codes; ideal macro accesses produce int64 results. The engine widens macro results to int64 before digital aggregation. Digital blocks use their operands' integer dtype without imposing an independent int64 input requirement. Ideal integer multiply-accumulate paths widen operands before arithmetic. The caller-selected storage width must represent the values and intermediates required by each operation, independently of modeled digital register widths.

## One physical access

A macro establishes the electrical boundaries and schedules the phases that use them. It owns peripheral sampling and the distinction between a boundary held across phases and a fresh boundary event. The array determines the electrical response of its cells and interconnect to those supplied conditions.

The array's operating point provides the terminal currents and voltages needed by the surrounding circuitry. The numerical iterations that find this point are internal to the electrical evaluation. They do not add phases to the physical schedule.

Shared boundaries follow the event lifetimes in [physical state](physical_state.md). Their establishment and phase-dependent activity contribute under [PPA accounting](ppa_accounting.md), regardless of how the numerical work is batched.
