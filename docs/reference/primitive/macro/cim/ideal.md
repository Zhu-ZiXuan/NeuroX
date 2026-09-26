# Ideal CIM macro

## Physical model

The ideal model preserves the family value domain, output code mapping, calibrated mode factors, and finite ADC resolutions while removing analog non-idealities. It computes an exact integer dot product for each WL plane and quantizes that dot when a finite resolution is requested.

## Governing equations

Let $M_p$ be one exact plane dot, $B$ the maximum ADC resolution, $b$ the active resolution, and $s_B$ the selected mode's configured rescale factor. Full-resolution quantization first forms

$$q_B=Q\!\left(\frac{M_p}{s_B}\right),$$

where evaluation uses floor and stochastic execution rounds upward with probability equal to the fractional part.

### Zero-point mapping

The full-resolution result is limited to the centered $B$-bit two's-complement range,

$$c_B=\operatorname{clamp}\left(q_B,-2^{B-1},2^{B-1}-1\right),$$

then arithmetically shifted to the active resolution,

$$\mathrm{code}_p=c_B\mathbin{\gg}(B-b).$$

Adding the fixed offset $2^{B-1}$ before truncation and removing the active-width offset gives the same centered code, spanning $[-2^{b-1},2^{b-1}-1]$.

### Sign-magnitude mapping

The sign is taken before quantization. The magnitude path computes

$$a_B=\operatorname{clamp}\!\left(Q\!\left(\frac{|M_p|}{s_B}\right),0,2^B-1\right),$$

then returns

$$\mathrm{code}_p=\operatorname{sgn}(M_p)\left(a_B\mathbin{\gg}(B-b)\right).$$

Its output spans $[-(2^b-1),2^b-1]$. The two raw representations of zero collapse to integer zero at the macro boundary.

Exact ideal execution returns $M_p$ unchanged with rescale factor one; finite-resolution execution applies the corresponding virtual quantization.

## Numerical method

The dot product uses fp32 contraction when its configured bound is below $2^{24}$, and int64 arithmetic otherwise. Quantization operates at $B$ bits before truncation to $b$ bits; integral scaled values remain deterministic under stochastic rounding.

## Symbols

The [family symbols](family.md#symbols) apply, with:

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $M_p$ | exact integer dot for plane $p$ | — | `plane_dot` |
| $q_B$ | factor-scaled full-resolution code before saturation | — | — |
| $c_B$ | saturated centered two's-complement code | — | — |
| $a_B$ | saturated full-resolution magnitude code | — | — |

## Assumptions, scope & validity

The fp32 contraction is exact only when the configured dot bound is below $2^{24}$ and operands obey the configured ranges and selection limit. The int64 path has no fp32 exactness assumption. Values outside the representable code domain saturate at the nearest endpoint.
