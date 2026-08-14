"""The rail boundary clamp of one DC solve: the driver role and its snapshot.

See Also:
    docs/internals/primitive/xbar/solver/clamp.md
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class ClampSnap(Protocol):
    """Required clamp-snapshot interface."""

    @property
    def v_ref__V(self) -> Tensor:
        """NOMINAL reference or zero-current clamp voltage, carrying no driver-owned perturbation.

        A concrete snap keeps any offset or noise draw in its own dedicated field(s), folded in by `solve_dc`.
        """
        ...


class ClampDcop(Protocol):
    """What a clamp driver's DC solution must expose.

    A clamp driver's DC solution carries these two quantities; a concrete
    module's dcop freely carries its own additional outputs beside them.
    """

    @property
    def v_clamp__V(self) -> Tensor:
        """Port clamp voltage at the solved operating point."""
        ...

    @property
    def dvclamp_di__MOhm(self) -> Tensor:
        """∂V_clamp/∂I, the clamp voltage's derivative against the port current."""
        ...


class ClampDriver[SnapT: ClampSnap, DcopT: ClampDcop](Protocol):
    """What a boundary clamp exposes to one DC solve.

    The whole structural role is a transfer law the solve evaluates at the
    port current inside its Newton loop, against a snap already sampled by the
    caller. Sampling that snap and delivering the clamp at the converged state
    are each a concrete circuit's own method.
    """

    def solve_dc(
        self,
        i_port__uA: Tensor,
        snap: SnapT,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> DcopT:
        """Solve the driver's DC state through its port.

        Args:
            i_port__uA: Port-output current the solve enters through.
            snap: Per-call snap, sampled by the concrete driver's own snapshot method.
            v_clamp_init__V: Optional warm-start hint.

        Returns:
            The module's own dcop, satisfying `ClampDcop`.
        """
        ...
