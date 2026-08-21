# CIM execution

One operator call becomes a fixed number of macro accesses. The mapping is shared between the unit holding the operator, the engine holding the schedule, its stages each holding one mapping and its inverse, and the macro holding one access; the axes are the seam between them, since whoever inserts an axis owes both the block that removes it and the cost multiplication over it.

## From one call to a count of accesses

Every extent below the unit is fixed when the tree is built, from the logical weight shape, the macro's port counts, and the weight layout's block width. One extent is not: the number of output planes a call carries, one for a linear operator and `H_out * W_out` for a convolution, which follows from the input resolution alone. The unit is therefore the single boundary at which a runtime shape enters the mapping — it converts the shape into a plane count and hands that number down, so nothing below it ever sees an operand layout and no configuration below it declares an input resolution.

What the engine reads off the macro is the logical face alone: the port counts it supplied at construction, the two value ranges, the selection limit, and the conversion metadata. It reads no row, column, digit, or readout structure, so a macro's internal encoding never reaches the placement arithmetic.

## The canonical axis layout

Between the engine's input face and the macro's, one tensor layout carries the whole schedule.

| Axis | Inserted by | Removed by |
|---|---|---|
| `M` output planes | the unit's operator lowering | the same operator's aggregation-undo |
| `Sx` input slices | the input-slice stage | that stage's shift adder |
| `Sw` weight slices | the weight-slice stage | that stage's shift adder |
| `Tc` contraction partitions | the placement stage | that stage's accumulator |
| `G` macro groups | the placement stage | flattened back into logical output order |
| `D` block slots | the placement stage | restored into output order, never reduced |
| `P` input phases | the input-activation stage | that stage's accumulator |

The layout is canonical rather than configuration-dependent: an axis a configuration does not use stays present at extent one, so every combination of stage choices traces one execution graph instead of one graph per combination. Each stage passes through every axis it does not own.

The macro's fabricated instance prefix reserves a slot for `M`, `Sx`, `Sw`, `Tc` and `G` in that order, the first two pinned to extent one; `D` and `P` sit ahead of the prefix as pure schedule axes. Cost is read off that alignment by one rule: a prefix slot carrying a real extent is parallel silicon, one physical copy per position, whereas a runtime axis broadcasting over an extent-one slot, or sitting ahead of the prefix entirely, is one copy used again in time. `Sw`, `Tc` and `G` are therefore silicon and `M`, `Sx`, `D` and `P` are time. One macro access serves each `(M, Sx, D, P)` point: the engine multiplies the macro's duration over exactly those four, and each point emits its own dynamic-energy event, while a parallel axis is billed once as the instances it is.

## Mapping and aggregation are paired

A stage owns a mapping together with the digital block that reverses it, rather than the two being configured independently. Two things follow. A mapping strategy cannot be combined with an aggregation that does not invert it, because neither half is separately selectable. And the digital silicon attaches to the operation it performs: a block's instance multiplicity is the shape of the tensor it actually reduces, which is why the contraction accumulator — folding its axis while the physical weight planes are still present — is instanced across those planes as well as across the macro groups.

## Who fixes the geometry

The weight-slice layout is interrogated before anything else is built. It answers with two numbers: the logical output width one weight block carries, and the number of macro planes one instance holds. The first is the block width the substrate-neutral planner partitions the logical outputs by; the second multiplies the macro's instance prefix. Changing the weight layout therefore moves the placement plan and the macro count together, which is why that stage's geometry is resolved ahead of the plan.

The planner itself carries no scheduling vocabulary. It partitions the contraction dimension, divides the logical outputs into blocks, balances those blocks across groups, and assigns each block a slot — and it never decides whether a group or a slot is realized in space or in time. The CIM side reads that decision into its own terms: contraction partitions become `Tc`, groups become `G`, block slots become `D`, and one block's contraction width becomes the local input length the activation stage then partitions into `P` groups under the macro's selection limit. Two of those axes become silicon and one becomes time, and nothing in the plan says which; that silence is what keeps the planner reusable under a different substrate.

## Integer exactness end to end

Every engine-side step is exact integer arithmetic: slicing and its radix-weighted shift-add are inverse operations, padding introduced by blocking encodes a true zero, and the final trim removes only padded positions. The single lossy step in the chain is the conversion inside one macro access, which is what makes an ideal-macro run a clean control: swapping the configured macro for its ideal twin leaves the placement, the slicing and the aggregation constructed identically, so the difference between the two runs is the analog access alone.

Two boundaries keep that exactness legible. The engine widens macro codes to the accumulation dtype once, at the analog-to-digital boundary, and no digital block converts dtype afterwards, so the accumulation domain is decided in one place. And the engine-backed unit is where a non-integer operand is refused — a generic operator base and an exact-integer reference unit impose no such gate, having no substrate that a float could be driven into.

## One access on a physical array

Below the macro face, a physical scheme decides how one access reaches the array's DC solve, and the two scan modes differ in kind rather than in mirror image. Each column owns its own ladders and boundary clamps with no equation coupling it to a neighbour, so the wire system is block-diagonal in the column axis. Every cell of one column, by contrast, shares that column's ladders, so moving which row is driven perturbs every node of every column.

A scheme that serializes along the column axis therefore folds its whole scan into one full-grid solve per plane: solving a group of columns and repeating over the groups gives per-column results identical to solving all of that plane's columns at once, the systems being the same blocks batched differently, and the total arithmetic is unchanged. The division of labour is the point rather than a saving — the scheme keeps its serial structure in its own representation, where transcoding, readout, billing and scheduling all need it, and the array is spared any notion of a serial slot. A scan along the row axis admits no such equivalence: each scanned row is a genuinely different network, so a row-organized scheme pays one solve per access. What one solve costs, and how the chunk loop bounds it, is [xbar_solve](xbar_solve.md).
