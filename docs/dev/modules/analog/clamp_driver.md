# Clamp-driver contract

`ClampDriver` is a structural Protocol describing the solver-facing behaviour of any voltage-clamped boundary driver.

The contract is the **behaviour**, not the concrete implementation:

- given a boundary current, solve the boundary operating point;
- report the clamp voltage and local small-signal sensitivity;
- keep any fabricated static state inside the owning circuit module.

Any concrete clamp driver that satisfies this contract — whether a feedback-loop TIA, an ideal voltage source, or a future variant — plugs into a generic DC solver through this Protocol without the solver having to import any concrete class.
