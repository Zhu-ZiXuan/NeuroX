# Transcoder

## Summary

The transcoder layer is the `Transcoder` ABC (`base.py`), which holds the shared positional state and the encoding-agnostic `decode`, plus one subclass per encoding (`true_form.py`, `complement.py`, `canonical.py`). It is a generic codec between integers and signed-digit strings over `(radix, digit_count)`: it converts one integer into a fixed-length positional digit string and reduces such a string back to its integer value. It models no hardware and no tile geometry — only the arithmetic of representing a signed integer in a chosen number system.

## Codec contract

A length-`digit_count` digit string $(d_0, d_1, \dots, d_{D-1})$ over a positional base (radix) $r \ge 2$, with $D$ = `digit_count` and $d_0$ the least-significant digit, denotes the integer formed by the radix-weighted sum

$$M = \sum_{i=0}^{D-1} d_i\, r^{i},$$

which is the single `decode` shared by all three encodings. The encodings differ only in the forward map (the digits each produces for a given integer $x$) and hence in the inclusive value range $[\,M_{\min}, M_{\max}\,]$ that round-trips losslessly. They agree on the radix and the positional weighting but differ in the digit alphabet:

- **true-form** — sign-magnitude: every non-zero digit carries the sign of the encoded integer; the magnitude is an ordinary base-$r$ decomposition. The digits are $d_i = \operatorname{sign}(x)\,\big(\lfloor |x| / r^{i}\rfloor \bmod r\big)$, giving the symmetric envelope

$$M \in \big[-(r^{D}-1),\ r^{D}-1\big].$$

- **radix-complement** — the low $D-1$ digits are unsigned in $\{0, \dots, r-1\}$ and only the most-significant digit folds into a signed alphabet, subtracting $r$ when it reaches or exceeds $\lceil r/2 \rceil$; this reduces to two's-complement at $r = 2$. The envelope is asymmetric,

$$M \in \Big[-\big\lfloor r/2 \big\rfloor\, r^{D-1},\ \ \big\lceil r/2 \big\rceil\, r^{D-1} - 1\Big].$$

- **canonical** — a non-adjacent-form-style signed-digit representation in which each digit lies in $\{-(r-1), \dots, r-1\}$; the round-trip contract bounds the magnitude by summing only the alternate positions,

$$M \in [-M_{\max},\ M_{\max}], \qquad M_{\max} = \sum_{j=0}^{\lceil D/2 \rceil - 1} (r-1)\, r^{\,D-1-2j}.$$

At $r = 2$ this is the non-adjacent form: no two consecutive positions are non-zero, so the alternate-position sum is exactly the maximum magnitude. This envelope is strictly tighter than the true-form bound for the same $(r, D)$.

The encoding choice selects which digit alphabet — and therefore which representable envelope — a digit string uses.

## Design decisions

- **Encoding is selected by a string discriminator, not an `isinstance` ladder.** `Transcoder` is a registry family keyed by the `Encoding` literal, so adding an encoding is one new file plus one registration decorator and never edits a shared factory; the dispatch substrate is [registry](../mixin/registry.md).
- **`decode` lives on the ABC; only `encode` and `value_range` are abstract.** The positional weighted sum is identical for every encoding (they differ only in the forward alphabet), so the shared reduction is written once on the base. Pushing it down to subclasses would duplicate it three ways and let them drift.

## Contracts & invariants

- **ABC observable surface.** A `Transcoder` exposes exactly: `encode(x, *, dim=-1) -> Tensor` (inserts a size-`digit_count` axis at `dim`), `decode(digits, *, dim=-1) -> Tensor` (removes that axis), and the `radix` / `digit_count` / `value_range` properties. `encode` and `decode` are mutual inverses *only within* `value_range`; outside it the forward map wraps and the round-trip is not recoverable - the value range is a caller contract, not an enforced clamp.
- **`encode` is shape-agnostic in `dim`.** The digit axis is inserted at the caller-chosen `dim`; the implementation must not assume a trailing axis. `decode` reduces whichever `dim` holds the digits and builds its positional-weight vector on the digit tensor's own device and dtype, so it stays correct across devices and never forces a host sync.
- **Validation is at construction.** `radix >= 2` and `digit_count >= 1` are checked in the base `__init__`; subclasses add no further construction validation.

## Numerical method

The encodings are exact integer arithmetic, evaluated digit-by-digit by repeated division and remainder by $r$ over the $D$ positions; there is no iteration to converge and no floating-point error. True-form takes the absolute value first and re-applies the sign per digit. Radix-complement post-folds only the top digit. Canonical carries a $+1$ into the next position whenever the current remainder rounds up, so the digit it emits at that position is the down-folded $d_i - r$; at $r = 2$ this carry reduces to the non-adjacent-form rule. The decode is the single weighted sum above.

## Performance & resources

- The encodings are `digit_count` division/remainder passes accumulated into a Python list, then one `torch.stack`. List accumulation (not in-place writes) keeps the unrolled loop fusable under `@torch.compile` at the caller. The work is integer-elementwise and negligible against the analog solve; there is no chunking or memory pressure at this layer.

## Gotchas

- **Value range is a contract, not a guard.** Encoding an out-of-range integer silently wraps or truncates - there is no error and no clamp. A caller must keep its inputs inside `value_range`; treating `encode`/`decode` as lossless for arbitrary integers is the anti-pattern.
- **Canonical carries across positions.** The canonical forward map mutates the running quotient with a carry while emitting each digit, so its per-digit step is not independent the way true-form's and complement's are; do not assume the three encodings share a digit loop body.

## Known limitations

- N/A.

---

- **Implementation**: `neurox/common/encoding/base.py`, `neurox/common/encoding/true_form.py`, `neurox/common/encoding/complement.py`, `neurox/common/encoding/canonical.py`
- **Tests**: `tests/test_transcoder.py`
- **Spec**: N/A — encoding is impl-only and has no reference page.
- **Decisions**: N/A — no ADR governs this module.
