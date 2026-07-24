# Ideal CIM macro

## Physical model

The ideal model preserves the family value domain and operating-point grid while
removing analog non-idealities. It computes an exact integer dot product for
each WL plane and optionally quantizes that plane independently.

## Governing equations

For a plane containing at most $A$ active rows,

$$M_{\max}=A\max|v|\max|x|.$$

At resolution $b\geq2$, the uniform signed rescale is

$$s=\frac{M_{\max}}{2^{b-1}-1}.$$

The deterministic evaluation-mode quantizer is

$$
\mathrm{code}_p=
\operatorname{clamp}\left(
\left\lfloor\frac{M_p}{s}\right\rfloor,
-2^{b-1},
2^{b-1}-1
\right).
$$

The lossless operating point bypasses this quantizer and returns $M_p$.

## Numerical method

Digits are combined with the LSB-first weights
$(1,r,\ldots,r^{D-1})$ before contraction over the full row extent. Zeroed
rows contribute nothing. Training mode replaces deterministic floor with
unbiased stochastic floor in code space.

## Symbols

The [family symbols](family.md#symbols) apply, with:

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $M_{\max}$ | maximum absolute plane dot | — | `_max_plane_dot_abs` |
| $M_p$ | exact integer dot for plane $p$ | — | `plane_dot` |

## Assumptions, scope & validity

The fp32 contraction is exact only when the configured plane-dot bound is below
$2^{24}$ and inputs obey the configured ranges and active-row limit. The int64
path has no fp32 exactness assumption.

---

- **Internals**: [ideal CIM macro](../../../../internals/primitive/macro/cim/ideal.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `IdealCimMacroConfig` (see `api`)
