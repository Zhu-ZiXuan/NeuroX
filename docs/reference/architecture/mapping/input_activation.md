# Input activation

Input activation enforces the macro's `max_active_num` limit without changing weight placement. It partitions each local weight-block input of length `L` into `P=ceil(L/max_active_num)` groups, masks all other positions to zero for each macro read, and accumulates the resulting `P` output codes.

Grouping and masking are pure data transformations. The unit evaluates the phase sum with its serial accumulator, including register-width wrap and circuit costs. Geometric block routing, weight slicing, and input-value slicing remain independent.
