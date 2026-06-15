# Signed-Digit Encodings

## Summary / role

A transcoder is the pure-math layer of the value-domain mapping: it converts one integer into a fixed-length positional signed-digit string and reduces such a string back to its integer value. It models no hardware — no tile geometry, no analog signal, no shape policy — only the arithmetic of representing a signed integer in a chosen number system. Three encoding policies differ in how the digit alphabet is signed, and therefore in the inclusive integer band each can losslessly round-trip. This document specifies the positional model, the three encodings, and the decode. The decomposition of a tensor into per-slice digit slots that feed a physical cell grid is the [slicer](../xbar/slicer/README.md)'s job.

## Physical model

There is no physical object here; the "model" is a number representation. A length-$D$ digit string $(d_0, d_1, \dots, d_{D-1})$ over a positional base (radix) $r \ge 2$ denotes the integer formed by the radix-weighted sum, with $d_0$ the least-significant digit. The three encodings agree on the radix and the positional weighting but differ in the digit alphabet:

- **true-form** — sign-magnitude: every non-zero digit carries the sign of the encoded integer; the magnitude is an ordinary base-$r$ decomposition.
- **radix-complement** — the low $D-1$ digits are unsigned in $\{0, \dots, r-1\}$ and only the most-significant digit folds into a signed alphabet; this reduces to two's-complement at $r = 2$.
- **canonical** — a non-adjacent-form-style signed-digit representation in which each digit lies in $\{-(r-1), \dots, r-1\}$; at $r = 2$ this is the non-adjacent form, where no two consecutive positions are non-zero.

The encoding choice is a string discriminator carried in configuration; it selects which alphabet (and therefore which representable envelope) a digit string uses.

## Governing equations

A digit string decodes by the positional weighted sum

$$M = \sum_{i=0}^{D-1} d_i\, r^{i},$$

which is the single decode shared by all three encodings. The encodings differ only in the forward map (the digits each produces for a given integer $x$) and hence in the inclusive value range $[\,M_{\min}, M_{\max}\,]$ that round-trips losslessly.

**True-form.** The digits are $d_i = \operatorname{sign}(x)\,\big(\lfloor |x| / r^{i}\rfloor \bmod r\big)$, giving the symmetric envelope

$$M \in \big[-(r^{D}-1),\ r^{D}-1\big].$$

**Radix-complement.** The low digits are the unsigned base-$r$ digits of $x$; the top digit $d_{D-1}$ is folded by subtracting $r$ when it reaches or exceeds $\lceil r/2 \rceil$, giving the asymmetric envelope

$$M \in \Big[-\big\lfloor r/2 \big\rfloor\, r^{D-1},\ \ \big\lceil r/2 \big\rceil\, r^{D-1} - 1\Big].$$

**Canonical.** Each digit lies in $\{-(r-1), \dots, r-1\}$, and the round-trip contract bounds the magnitude by summing only the alternate positions,

$$M \in [-M_{\max},\ M_{\max}], \qquad M_{\max} = \sum_{j=0}^{\lceil D/2 \rceil - 1} (r-1)\, r^{\,D-1-2j}.$$

At $r = 2$ this is the non-adjacent form: no two consecutive positions are non-zero, so the alternate-position sum is exactly the maximum magnitude. This envelope is strictly tighter than the true-form bound for the same $(r, D)$.

An integer outside an encoding's value range does not round-trip: the forward map silently wraps or truncates it. The value range is therefore a contract a caller must respect, not a clamp the transcoder enforces.

## Numerical method

The encodings are exact integer arithmetic, evaluated digit-by-digit by repeated division and remainder by $r$ over the $D$ positions; there is no iteration to converge and no floating-point error. True-form takes the absolute value first and re-applies the sign per digit. Radix-complement post-folds only the top digit. Canonical carries a $+1$ into the next position whenever the current remainder rounds up, so the digit it emits at that position is the down-folded $d_i - r$; at $r = 2$ this carry reduces to the non-adjacent-form rule. The decode is the single weighted sum above.

## Noise & non-idealities

N/A - the transcoder is exact integer arithmetic; it introduces no noise or non-ideality. Quantization and analog error enter only downstream in the device, circuit, and ADC models, not in the value-domain mapping.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `radix` | positional base $r$ ($\ge 2$) of the digit representation | — | Design |
| `digit_num` | number of digits $D$ ($\ge 1$) in the string | — | Design |
| `encoding` | encoding policy discriminator (`true_form` / `complement` / `canonical`) | — | Design |

Provenance terms: [parameter_provenance](../../parameter_provenance.md). The runtime integer tensor being encoded is an input, not a parameter. Schema: `[mapper]` config (see config reference).

## Assumptions, scope & validity

- Inputs are integer-valued; the transcoder does not quantize a real number, only re-represent an integer.
- An input must lie within the selected encoding's value range to round-trip; out-of-range inputs wrap or truncate silently (no clamp).
- The transcoder fixes the radix and digit count but holds no notion of tile shape, padding, or cell geometry - those are decided above it.

TODO (domain author): state any additional validity limits (e.g. the largest $(r, D)$ for which the integer dtype does not overflow the radix weights).

## Validation

TODO - link the round-trip and value-range evidence (`tests/test_transcoder.py` exercises encode/decode round-trips and the value-range bounds) once a `validation/mapper` page exists.

## References

TODO (domain author): cite the canonical-signed-digit / non-adjacent-form representation and the radix-complement number system.

---

- **Internals**: [transcoder internals](../../../internals/mapper/transcoder/encodings.md)
- **Validation**: TODO - `validation/mapper` (not yet written)
- **Configuration**: [config reference](../../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
