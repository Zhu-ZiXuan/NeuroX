# ADR-0005: MOSFET is a polarity-parameterized electrical primitive

## Status

Accepted

## Context

[ADR-0002](ADR-0002-nmos-is-a-pure-electrical-primitive.md) reduced the transistor to a pure electrical primitive — electrical and mismatch parameters plus `W__um`/`L__um` design kwargs, with layout parasitics owned by the consuming circuit. That primitive was n-channel only.

The simulator now needs a p-channel device (pseudo-resistor loads, complementary clamps). Two near-duplicate primitives — one n-channel, one p-channel — would diverge over time and double the physics surface that has to stay correct.

The two polarities are not independent physics. An n-channel and a p-channel transistor obey the same square-law overdrive equations under a single sign flip applied to the gate-overdrive and drain-source-current orientation. SPICE/BSIM already encodes the asymmetry this way: it hands `VTH0` *signed* (negative for a typical p-channel enhancement device) while mobility `U0` stays a *positive* magnitude for both polarities.

## Decision

A polarity-parameterized `MOSFET` base (`neurox/device/mosfet.py`) carries all device physics. A class attribute `polarity` (p) selects the channel: concrete `NMOS` sets `polarity = +1` and `PMOS` sets `polarity = -1`, and they specialize *only* that attribute. The type identifiers for the shared data classes are renamed off the `NMOS` prefix to the primitive: `MOSFETConfig`, `MOSFETPolicy`, `MOSFETSnap`, `MOSFETDCOP`.

The physics enters polarity once and is designed so the polarity factor does not leak into the derived contracts:

- **(1) `beta`/`mu0` stay positive magnitudes.** The polarity p multiplies the gate overdrive ($V_{\mathrm{ov},s} = p\,(V_g - V_s - V_{th})$, $V_{\mathrm{ov},d} = p\,(V_g - V_d - V_{th})$) and the drain-source current ($I_{ds} = \tfrac{1}{2}\,p\,\beta\,(v_s^2 - v_d^2)$, with $v_{s,d} = \operatorname{softplus}_\lambda(V_{\mathrm{ov},\{s,d\}})$). Because the overdrive carries one p and the current carries one p, the *terminal* partials see $p^2 = 1$ and have no explicit polarity factor: $\partial I/\partial V_d = \beta\,v_d\,\sigma_d \ge 0$ and $\partial I/\partial V_s = -\beta\,v_s\,\sigma_s \le 0$ hold for *both* polarities. The `did_dvd >= 0` / `did_dvs <= 0` monotonicity contract is therefore polarity-free, and the beta-mismatch sigma ($\sigma_\beta = \beta_{\mathrm{nom}}\,A_{\beta,\mathrm{rel}}/\sqrt{W L}$) needs no `abs` since $\beta_{\mathrm{nom}}$ is positive.

- **(2) `vth0` stays signed, and its sign is the enhancement/depletion flavor — not the polarity.** Polarity already lives in p; `vth0` is free to be negative for a depletion device of either channel. Tying the threshold sign to the polarity would make depletion devices unrepresentable.

- **(3) Config, policy, snap, and DCOP are shared and polarity-free.** `MOSFETConfig`/`MOSFETPolicy`/`MOSFETSnap`/`MOSFETDCOP` carry no n/p-specific fields; there are no `NMOS`- or `PMOS`-specific data classes.

- **(4) Polarity is a code/architecture decision, not a config field.** The consuming circuit instantiates `NMOS` or `PMOS` directly — the access transistor or pseudo-resistor *is* an n-channel device by design, and that is a structural fact of the circuit, not a tunable. Config files keep the n/p distinction through preset and section naming (`[nmos_28_rvt]`, `[hardware.nmos_config]`) and through consumer role field names (`access_nmos_*`, `pseudo_nmos_*`, `v_nmos_bias__V`), all of which are kept unchanged.

- **(5) The `I_ds` sign convention is drain-to-source positive.** A p-channel device in normal conduction therefore carries a negative `I_ds`.

## Consequences

Positive:

- One physics core serves both polarities; correctness is maintained in a single place.
- `PMOS` is available with no duplicated equations.
- Mobility stays an honest positive magnitude, matching SPICE/BSIM `U0`.
- Consumer churn is minimal: only the shared *type names* change (`NMOSConfig` -> `MOSFETConfig`, etc.); every *role* name that denotes a specific designed n-channel transistor (`nmos_config`, `self.nmos`, `access_nmos_*`, `pseudo_nmos_*`, `v_nmos_bias__V`, and the TOML presets/sections) is preserved, so the design intent "this transistor is an NMOS" is not erased.

Tradeoff:

- N/P consistency between the instantiated class (`NMOS` vs `PMOS`) and the signed `vth0` on the config card is a naming convention, not machine-enforced. A `PMOS` built from a card with a positive `vth0`, or an `NMOS` from a negative one, is not rejected by the type system.

## Relationship

This ADR generalizes [ADR-0002](ADR-0002-nmos-is-a-pure-electrical-primitive.md): `NMOS` is now the n-channel specialization of the `MOSFET` primitive. ADR-0002's core decision still holds for `MOSFET` — it remains a pure electrical primitive with no layout parasitics, and `W__um`/`L__um` stay init kwargs while parasitic lumping is owned by the consuming circuit.
