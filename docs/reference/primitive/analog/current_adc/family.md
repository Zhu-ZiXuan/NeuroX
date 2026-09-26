# Single-ended current ADC family

## Shared conventions

A current ADC digitizes a single-ended, non-negative magnitude current $I_{\mathrm{in}}$ [uA] into one **unsigned** integer code. The converter commits to the current domain: the input, every reference level, and every noise term are expressed in microamperes.

The reference values $I_{\mathrm{ref}}$ are a runtime input, ascending along a trailing tap axis. The resolution $b$ and the injected references fully determine one conversion; an operating-mode identity is not part of the transfer relation.

The number of reference values consumed by one conversion is topology-specific rather than a family law.

The resolution $b$ sets the number of code levels; the maximum resolution $b_{\max}$ is fixed for a given ADC. Every call supplies an integer `active_bits` in `[1, bits]`. The references describe physical wiring, not the requested resolution, so a conversion at $b < b_{\max}$ is realized inside the converter by running fewer decision cycles over the same taps.

## Governing laws

The family quantization is a **monotone** mapping of the magnitude input against the injected reference levels $\{I_{\mathrm{ref},c}\}$, all carrying microampere units. A conversion returns a **raw** unsigned code in

$$\mathrm{code} \in [0,\ 2^{b}-1],$$

the count of reference levels the input reaches or exceeds, resolved to $b$ bits. Because the input is a non-negative magnitude, there is no zero-code shift: the raw unsigned code is the direct output.

Bit width nests: for a deterministic member the code at resolution $b$ is the code at $b_{\max}$ right-shifted,

$$\mathrm{code}_{b} = \big\lfloor \mathrm{code}_{b_{\max}} / 2^{\,b_{\max}-b} \big\rfloor,$$

so lowering $b$ coarsens the bin without moving the transfer.

## Noise & non-idealities

Quantization against the reference levels is intrinsic to every member; all further non-idealities are topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $I_{\mathrm{in}}$ | single-ended magnitude input current | uA | `i_in__uA` |
| $b$ | active conversion resolution, per call | — | `active_bits` |
| $b_{\max}$ | physical output bit width | — | `bits` |
| $I_{\mathrm{ref}}$ | per-call injected reference levels, $[\ldots,\ n_{\mathrm{ref}}]$ with the taps last | uA | `i_refs__uA` |
| $n_{\mathrm{ref}}$ | reference count the converter's circuit takes | — | member-defined |

## Assumptions, scope & validity

The input is a single-ended nonnegative magnitude.

Physical operating limits require characterization of the concrete converter.
