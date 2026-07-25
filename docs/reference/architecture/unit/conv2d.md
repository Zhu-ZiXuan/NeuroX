# Conv2d mapping

The conv2d operator dimension: how a unit lowers `F.conv2d` onto its matmul-shaped substrate. The mapping is Toeplitz / input-stationary — the kernel is unrolled at program time into a sparse weight matrix whose zeros perform the window selection, and each substrate plane carries an input strip covering a group of consecutive output positions of one output row. The classic im2col lowering is the degenerate single-window case of the same structure.

## Governing laws

**Exact replica.** Over the accepted integer value domain the lowered operator computes exactly `F.conv2d` (grouped convolution out of scope): for input with trailing $[C_{\mathrm{in}}, H, W]$, kernel $(k_h, k_w)$, stride $(s_h, s_w)$, padding $(p_h, p_w)$, dilation $(d_h, d_w)$,

$$H_{\mathrm{out}} = \left\lfloor \frac{H + 2p_h - d_h(k_h-1) - 1}{s_h} \right\rfloor + 1,$$

analogously $W_{\mathrm{out}}$. The lowering arithmetic — strip gather, Toeplitz contraction, fold, trim, bias add — is exact integer arithmetic by construction; the only deviation is the substrate's own non-ideality.

**Strip geometry.** One substrate plane covers $W_g$ consecutive output positions of one output row. With the dilated kernel width

$$k_w^{\mathrm{eff}} = (k_w - 1)\, d_w + 1,$$

the strip spans

$$W_{\mathrm{strip}} = k_w^{\mathrm{eff}} + (W_g - 1)\, s_w$$

input columns and persists the $k_h$ used input rows, giving the lowered matmul dims

$$K' = C_{\mathrm{in}}\, k_h\, W_{\mathrm{strip}}, \qquad N' = W_g\, C_{\mathrm{out}}.$$

**Window-group rule.** $W_g$ is a mapping policy derived from the substrate geometry, never a configuration field: the maximal $W_g$ satisfying both

$$K' \le N_{\mathrm{in}} \qquad \text{and} \qquad W_g\, C_{\mathrm{out}} \le N_{\mathrm{out}},$$

floored at $W_g = 1$. At the floor even the single-window strip may exceed one tile; the generic engine tiling then splits $K'$/$N'$ as usual.

**Toeplitz placement law.** For window $g \in [0, W_g)$ and output channel $n$, kernel entry $\mathrm{weight}[n, c_i, i, j]$ lands at

$$c = g\, C_{\mathrm{out}} + n, \qquad r = (c_i\, k_h + i)\, W_{\mathrm{strip}} + (g\, s_w + j\, d_w)$$

of the $(N', K')$ matrix; all other entries are 0. The zeros — dilation gaps and inter-window gaps — are the row selection itself: no per-window row masks exist anywhere in the mapping, and the substrate must be able to carry the value 0 on the weight side.

**Strip gather.** For output row $h_o \in [0, H_{\mathrm{out}})$ and strip segment $t \in [0, T_{\mathrm{seg}})$ with $T_{\mathrm{seg}} = \lceil W_{\mathrm{out}} / W_g \rceil$, the plane reads input rows $h = h_o s_h + i\, d_h - p_h$ for $i \in [0, k_h)$ and input columns $w = t\, W_g s_w - p_w + [0, W_{\mathrm{strip}})$; out-of-bounds positions are zero-filled. Window $g$ of segment $t$ computes output column $w_o = t\, W_g + g$: its tap $j$ reads strip offset $g s_w + j d_w$, i.e. input column $w_o s_w - p_w + j d_w$ — exactly `F.conv2d` indexing. The gather keeps $[H_{\mathrm{out}}, T_{\mathrm{seg}}]$ as serial axes; surplus windows of the last segment ($w_o \ge W_{\mathrm{out}}$) read only zero-filled columns, compute valid padded-input results, and are trimmed at fold.

**Fold.** The substrate output unflattens $N' \to (W_g, C_{\mathrm{out}})$ (window outer, channel inner — the column law above), assembles the output columns as $(H_{\mathrm{out}}, T_{\mathrm{seg}} \cdot W_g)$, trims to $W_{\mathrm{out}}$, and arranges to trailing $[C_{\mathrm{out}}, H_{\mathrm{out}}, W_{\mathrm{out}}]$; the per-channel integer bias is added exactly once per real output element, after the trim.

**Degenerate case.** $W_g = 1$ gives $W_{\mathrm{strip}} = k_w^{\mathrm{eff}}$, $K' = C_{\mathrm{in}} k_h k_w^{\mathrm{eff}}$, $N' = C_{\mathrm{out}}$: one window per plane — the im2col lowering as a special case of the Toeplitz structure (with dilation gaps still resolved by the matrix zeros).

**Drive semantics.** Every substrate input phase is independently driven. Phase
serialization is the substrate's generic mechanism; the conv2d mapping contains
no phase-splitting logic of its own.

## Noise & non-idealities

The mapping adds no non-ideality of its own: gather, placement, fold, trim, and bias add are exact. Every deviation enters through the substrate reads, per the [family contract](family.md#noise-non-idealities).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $C_{\mathrm{in}}, C_{\mathrm{out}}$ | input / output channels | — | `w_logical_shape` |
| $k_h, k_w$ | kernel extent | — | `w_logical_shape` |
| $s_h, s_w$; $p_h, p_w$; $d_h, d_w$ | stride; padding; dilation | — | `stride`, `padding`, `dilation` |
| $k_w^{\mathrm{eff}}$ | dilated kernel width | — | — |
| $W_g$ | output positions per plane (window group) | — | derived at construction |
| $W_{\mathrm{strip}}$ | input columns per strip | — | derived at construction |
| $K', N'$ | lowered contraction / output dims | — | engine logical shape |
| $T_{\mathrm{seg}}$ | strip segments per output row | — | — |
| $N_{\mathrm{in}}, N_{\mathrm{out}}$ | macro logical input / output capacity | — | `engine.input_num`, `engine.output_num` |
| $b$ | integer bias vector (length $C_{\mathrm{out}}$) | — | `int_bias` |

The value-domain and ADC-surface symbols are in [family](family.md#symbols).

## Assumptions, scope & validity

- The contract is defined for integer operands within the published value ranges; requantization lies outside the unit.
- The weight value range must cover 0 unconditionally (the Toeplitz zeros are stored values); the activation range must cover 0 whenever zero-filled activations occur — non-zero padding, or $W_g > 1$ (strip right-fill and last-segment surplus windows).
- Grouped convolution is out of scope.

## References

TODO.

---

- **Internals**: [conv2d operator internals](../../../internals/architecture/unit/conv2d.md), [Conv2dCimUnit internals](../../../internals/architecture/unit/cim/conv2d.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../api/README.md)
