# Scan organizations

A tile access drives one boundary axis with the input and steps the other one position at a time. Which axis does which is the array's **scan organization**, carried by `XbarArray1t1rOperationMode` and fixed at construction. Two archetypes exist:

- `WL_IN_BL_SCAN` (`wl_in_bl_scan`) — the word lines carry the held input and the bit-line boundary is scanned.
- `BL_IN_WL_SCAN` (`bl_in_wl_scan`) — the bit-line boundary carries the held input and the word lines are scanned.

## The two organizations are not mirror images

Columns are independent and rows are not, and that single structural fact makes the two scans behave differently under serialization.

Each column owns its own bit-line and source-line ladder, its own boundary clamps, and no equation coupling it to a neighbour: the wire system is block-diagonal in the column axis. Every cell of one column, by contrast, sits on that column's two shared ladders, so moving which row is driven perturbs every node of every column.

A scan along the column axis therefore steps an axis the solve already treats as independent. Whether the columns are presented one group at a time or all at once changes nothing the solve can observe, so the whole scan folds into a single solve over the full physical grid. A scan along the row axis changes the drive pattern the ladders see, so each scanned position is a genuinely different network and needs its own solve.

## Flattening equivalence

Serialization commutes with solving on the column axis. Solving a group of columns, repeated over the groups of one plane, and solving all of that plane's physical columns in one call produce identical per-column results — the systems are the same block-diagonal blocks, merely batched differently — and the total arithmetic is unchanged, since the group count times the columns per group is the physical column count.

The consequence is a division of labour rather than a saving: a column-serialized scheme keeps its serial structure in its own representation, where transcoding, readout, billing, and latency all need it, and hands the array one full-grid solve per plane. The array is spared any notion of a serial slot, and the scheme is spared the per-group solve overhead.

No such equivalence exists on the row axis, so a row scan is a real per-access solve and a scheme organized that way pays one solve per scanned row.

## What the mode selects

Energy evaluation, and nothing else. Programming writes the same physical state grid and a solve settles the same node network under either organization, so neither reads the mode; the mode picks which of the two per-solve billing functions the measurement calls. Each of the two is the complete account of one access — the four node totals of every cell, each at its own displacement — so what the two differ on is only the rest state each access departs from. The billing law itself — the supply-draw atom, the rail split, and the one-charging-leg rule — is in [capacitive energy accounting](../../../reference/primitive/physics.md).

- `wl_in_bl_scan` — nothing is held between accesses, so every node rests at ground, an access is one complete excursion, and there is no hold to establish.
- `bl_in_wl_scan` — the conduction path rests at its ideal held level while the control line rests at ground, so an access bills the displacement away from that rest level plus its share of establishing the hold.

The rest state is declared, not solved: both functions read the two boundary levels off the clamp snaps rather than off the converged state, and one node deeper each cell's internal access node rests at its own bit-line level — the storage element sits on the bit-line side of an access device orders of magnitude less conductive when off, so the off divider parks the node there.

## Hold contract

The held organization carries one contract the billing depends on: **one hold covers exactly one full row scan**, so the establishment cost is divided by the row count and a complete scan bills exactly one establishment. The contract is an assertion about the schedule the owning scheme runs, not something the array can verify — the array sees one access at a time and has no view of the scan it belongs to.

A scheme whose hold spans a different number of accesses does not reach for a new mode or a new divisor argument. It overrides the per-chunk measurement, which is where the billing functions are called, and states its own amortization there.

## See also

- [capacitive energy accounting](../../../reference/primitive/physics.md)
- Implementation: `neurox/primitive/xbar/array/_1t1r.py`
