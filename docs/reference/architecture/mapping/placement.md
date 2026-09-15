# Matrix tiling and sharing

Tiling partitions the contraction dimension and logical output dimension into capacity-bounded weight blocks. Partial dot products from contraction tiles are accumulated; output tiles are concatenated in logical order.

Input-slot sharing is optional. It packs short input blocks into disjoint input intervals of a macro and schedules those intervals serially. Without sharing, each output block occupies its own macro group. Sharing changes physical multiplicity and execution duration without changing the logical matrix operation.

Each scheduled block carries its effective output count, including zero for absent blocks. Precision slicing translates that count to logical macro ports. The macro maps those ports to its own lane and scan arrangement.
