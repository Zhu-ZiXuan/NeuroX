# Voltage ADC family

Every member digitizes a differential voltage into a raw unsigned integer code.

## Shared conventions

An ADC digitizes a positive leg $V^{+}$ against a negative leg $V^{-}$. Both
legs, every noise term, and the code boundaries are expressed in volts. An
externally supplied reference sets the full-scale range, and the resolution $b$
sets the number of code levels within it.

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

---

- **Internals**: [voltage ADC base](../../../../internals/primitive/analog/voltage_adc/base.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `DifferentialVoltageAdcConfig` (see `api`)
