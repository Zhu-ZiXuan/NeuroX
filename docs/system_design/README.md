# System design

The software contracts that span components, one page per maintainer question.

- [construction](construction.md) — how a configuration file set becomes a live module tree, and what selects the implementation behind each polymorphic slot
- [physical_state](physical_state.md) — where a physical module's tensor state lives, when each value is drawn, and what the fabricate and program lifecycle settles
- [ppa_accounting](ppa_accounting.md) — how an emitter lays out the tensor it bills, how the measurement reduces it without knowing the emitter, and what a mis-shaped bill costs
- [compile](compile.md) — which side of the eager / compiled boundary a function sits on, why the boundary is drawn around the DC solve, and what a crossing costs
- [cim_execution](cim_execution.md) — how one operator call becomes a fixed count of macro accesses, and which component inserts, removes, and pays for each axis on the way
- [xbar_solve](xbar_solve.md) — which component owns which part of one array DC solve, why the chunk loop bounds memory without moving a number, and what probing the iteration costs
- [nonideality_kernels](nonideality_kernels.md) — which half of a perturbation the shared kernel owns and which half the calling model owns, and how an enable flag reaches the graph
