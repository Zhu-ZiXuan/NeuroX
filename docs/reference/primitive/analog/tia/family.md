# TIA family

Each concrete TIA is a bit-line (BL) clamp that holds its input at a virtual-ground reference and absorbs the column boundary port current, reporting the clamp voltage and its small-signal sensitivity $\partial V_{\mathrm{clamp}}/\partial I_{\mathrm{port}}$.

## Shared conventions

The virtual-ground reference $V_{\mathrm{ref}}$ is externally supplied per call, not a family constant; each member holds its input at $V_{\mathrm{ref}}$ and absorbs the boundary port current, converting it into a clamp voltage. The clamp is non-ideal: the clamp voltage responds to the port current through a finite sensitivity, the ideal zero-input-impedance clamp being only the limiting case.

## Governing laws

The clamp transfer function maps the boundary port current to the clamp voltage at the BL port,

$$V_{\mathrm{BL,CL}} = \operatorname{TIA}(I_{\mathrm{BL,port}}),$$

with small-signal sensitivity

$$\frac{\partial V_{\mathrm{BL,CL}}}{\partial I_{\mathrm{BL,port}}}$$

(in MOhm), zero for an ideal clamp (the infinite-gain, zero-input-impedance limit) and non-zero for a finite-gain TIA. $V_{\mathrm{ref}}$ is the virtual-ground level the input is held at; the concrete form of $\operatorname{TIA}(\cdot)$ is topology-specific.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_clamp__V` |
| $V_{\mathrm{ref}}$ | virtual-ground reference level | V | `snapshot(v_ref__V=...)`, `*Snap.v_ref__V` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | `i_port__uA` |

## Assumptions, scope & validity

The clamp transfer function is assumed monotone in the port current, the family's contribution to a unique boundary operating point.

TODO (domain author): the input-current range over which the virtual-ground clamp holds, common to the family.

## References

TODO.

---

- **Internals**: [tia base internals](../../../../internals/primitive/analog/tia/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `TiaConfig` (see `api`)
