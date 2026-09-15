# Weight slicing

Weight slicing represents each weight using positional digits:

$$W=\sum_{s=0}^{S_w-1}w_sR_w^s.$$

Numerical decomposition precedes geometric tiling. The tiling strategy determines
both tile boundaries and slice placement. Two strategies support decomposed and
unsliced values:

| Placement | Physical layout | Logical outputs per tile |
| --- | --- | --- |
| Inter-plane | one macro plane per slice | physical output count |
| Intra-port | adjacent output ports carry one weight's slices | floor(physical output count / $S_w$) |

Inter-plane placement spends $S_w$ times as many macro instances. Intra-port
placement keeps one macro plane but reduces useful logical output width and may
leave trailing ports idle. With one slice, both placements preserve the full output
capacity.

Tile recovery restores separate slices and logical outputs together. Numerical
reconstruction then combines the positional contributions. A
configured shift adder supplies register-width arithmetic and circuit costs;
without it, reconstruction is a functional weighted sum.
