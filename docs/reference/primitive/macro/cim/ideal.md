# Ideal CIM macro

## Physical model

The ideal model preserves the family value domain and operating-point grid while
removing analog non-idealities. It computes an exact integer dot product for
each WL plane and, at each quantization mode, quantizes that dot against the
mode's declared conversion window.

## Governing equations

For plane dot $M_p$ and the inclusive window $[M_{\min}, M_{\max}]$ that the
quantization mode selects, the window holds

$$N_{\mathrm{q}}=M_{\max}-M_{\min}+1$$

quantization targets, so at resolution $b\geq1$ the step is

$$\Delta=\frac{N_{\mathrm{q}}}{2^{b}},$$

which need not be an integer. The unsigned reading a converter performs is

$$
\mathrm{code}_u=\operatorname{clamp}\left(
\left\lfloor\frac{M_p-M_{\min}}{\Delta}\right\rfloor,
0, 2^{b}-1
\right).
$$

The macro subtracts the window zero code, which the window shape fixes,

$$
z=\begin{cases}0 & M_{\min}=0\\ 2^{b-1} & M_{\min}<0\end{cases}
$$

and returns the signed code $\mathrm{code}_p=\mathrm{code}_u-z$, equivalently

$$
\mathrm{code}_p=\operatorname{clamp}\left(
\left\lfloor\frac{M_p}{\Delta}\right\rfloor,
-z, 2^{b}-1-z
\right).
$$

A signed code therefore spans $[0, 2^{b}-1]$ for an unsigned window and
$[-2^{b-1}, 2^{b-1}-1]$ for a mid-zero one. Bit widths nest: the unsigned code
at $b$ is the unsigned code at the maximum resolution right-shifted by the
width difference. The lossless operating point requests no resolution at all
and bypasses this quantizer, returning $M_p$ unmodified.

## Numerical method

The logical matrix is contracted directly with the input vector — fp32
`einsum` when the configured dot bound (`_max_plane_dot_abs`) is below
$2^{24}$, int64 multiply-reduce otherwise. Zeroed positions contribute
nothing. The conversion evaluates entirely in int64: $\mathrm{code}_u$ is a
floor division of $(M_p-M_{\min})\cdot 2^{b}$ by $N_{\mathrm{q}}$, never forming
the fractional $\Delta$; floor division rounds toward $-\infty$. Training mode
adds a uniform integer jitter on $[0, N_{\mathrm{q}}-1]$ to the floor numerator
before dividing,
an unbiased stochastic rounding of the floor remainder that leaves every
on-grid dot, the window zero included, on its deterministic code. Evaluation
mode takes the bare floor and is deterministic.

## Symbols

The [family symbols](family.md#symbols) apply, with:

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $M_p$ | exact integer dot for plane $p$ | — | `plane_dot` |
| $M_{\min}, M_{\max}$ | inclusive conversion window bounds of the selected mode | — | `quantization_input_ranges` |
| $N_{\mathrm{q}}$ | quantization targets held by the window | — | — |
| $\Delta$ | quantization step of the window at resolution $b$ | — | — |
| $z$ | window zero code | — | — |

## Assumptions, scope & validity

The fp32 contraction is exact only when the configured dot bound is below
$2^{24}$ and operands obey the configured ranges and selection limit. The
int64 path has no fp32 exactness assumption. Every window is canonical —
unsigned $[0, M_{\max}]$ or mid-zero $[-m, m-1]$ — enforced at config
validation, which is what puts the zero point on a bin edge, and hence $z$ on
an integer, at every supported bit width. A window with
$N_{\mathrm{q}} > 2^{b}$ is a legal lossy operating point; a dot outside the
window clips to the nearest end code.

The ideal counterpart of a sign-magnitude member is not bit-exact against it
at any finite resolution: a mid-zero window carries one phantom bottom level
and a single zero, whereas a sign-magnitude encoding has no bottom level and a
double zero. That code-grid difference is part of what an ideal comparison
measures, not an artifact to be corrected for.

External tiled-ADC formulations often record a half-range full-scale bound; a
window here is the full inclusive range, so an external full-scale $F$
corresponds to the mid-zero window $[-F, F-1]$.
