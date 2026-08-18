# Input activation

Input activation enforces the macro's `max_active_num` limit without changing
weight placement. It partitions each local weight-block input of length `L`
into `P=ceil(L/max_active_num)` groups, masks all other positions to zero for
each macro read, and accumulates the resulting `P` output codes.

The stage owns the `P`-axis serial accumulator through
`InputActivationStageConfig`. Geometric block routing, weight slicing, and
input-value slicing remain independent.
