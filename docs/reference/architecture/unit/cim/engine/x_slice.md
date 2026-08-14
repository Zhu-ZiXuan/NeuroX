# Input slicing

Input slicing serializes a logical input value across `Sx` macro reads.
`SerialXSliceStage` decomposes the value into positional digits with radix

$$R_a=x_{\max}-x_{\min}+1,$$

where `[x_min,x_max]` is one macro read's accepted input range. Its paired
shift adder reconstructs

$$X=\sum_{s=0}^{S_x-1}x_sR_a^s.$$

`DirectXSliceStageConfig` selects no decomposition and retains a structural
`Sx=1` axis. `SerialXSliceStageConfig` selects a positive `x_slice_num` and
owns the corresponding `Sx` shift-adder configuration.

`Sx` is purely a time axis: it increases macro reads but not the programmed
macro instance count. The stage publishes the resulting logical
`x_value_range`.

---

- **Internals**: [input-slice internals](../../../../../internals/architecture/unit/cim/engine/x_slice.md)
- **Engine**: [CIM engine](family.md)
