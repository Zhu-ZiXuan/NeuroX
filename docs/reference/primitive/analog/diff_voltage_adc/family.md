# Differential voltage ADC family

Every member digitizes a differential voltage into a raw unsigned integer code.

## Shared conventions

An ADC digitizes a positive leg $V^{+}$ against a negative leg $V^{-}$. Both
legs, every noise term, and the code boundaries are expressed in volts. The
reference values $V_{\mathrm{ref}}$ are supplied per call along a trailing tap
axis and set the full-scale range; the resolution $b$ sets the number of code
levels within it. No member holds its own references, and no operating-mode
identity reaches a converter: the owner names a mode to its reference source and
receives the matching taps already selected.

How many reference values one conversion takes is the member's circuit property,
not a family law. A topology that takes one full-scale reference and divides it
internally receives one tap; a comparator bank receives its whole threshold set.

## Governing laws

The family quantization is monotone against an ordered boundary set $\{B_c\}$:

$$
\mathrm{code} =
\operatorname{clamp}\left(
\operatorname{bucketize}(V^{+}-V^{-},\{B_c\}),
0,
n_{\mathrm{codes}}-1
\right).
$$

The returned code is the raw bucket index. The corresponding signed physical
magnitude follows

$$
M_{\mathrm{ideal}}
\approx
(\mathrm{code}-z)\cdot\mathrm{rescale\_factor},
\qquad
\mathrm{rescale\_factor}>0,
$$

where $z$ is the topology-specific zero-point offset. The conversion itself does
not apply this affine recovery.

For uniform boundaries, $\mathrm{LSB}=\mathrm{FSR}/2^b$ and
$n_{\mathrm{codes}}=2^b$. A symmetric topology has $z=2^{b-1}$ and therefore
represents the signed range $[-2^{b-1},2^{b-1}-1]$ after zero-point removal.

## Noise & non-idealities

Quantization is intrinsic to every member; further non-idealities are
topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V^{+},V^{-}$ | differential input legs | V | `v_pos__V`, `v_neg__V` |
| $V_{\mathrm{ref}}$ | per-call injected reference taps, $[\ldots,\ n_{\mathrm{ref}}]$ with the taps last | V | `v_refs__V` |
| $n_{\mathrm{ref}}$ | reference count the member's circuit takes | — | member-defined |
| $b$ | ADC resolution | — | `bits` |
| $\mathrm{FSR}$ | full-scale input range | V | derived |
| $\mathrm{LSB}$ | uniform code step | V | derived |
| $B_c$ | code boundary at index $c$ | V | derived |
| $n_{\mathrm{codes}}$ | number of raw code buckets | — | derived |
| $z$ | zero-point offset | — | `zero_offset(bits)` |

## Assumptions, scope & validity

The differential input is two-sided and $b\geq1$. At $b=1$, the raw codes are
$\{0,1\}$ with symmetric zero point $z=1$ and signed range $[-1,0]$.

TODO (domain author): state the operating envelope over which the monotone
boundary model remains valid.
