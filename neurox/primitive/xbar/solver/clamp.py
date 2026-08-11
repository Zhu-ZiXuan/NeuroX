"""The rail boundary clamp of one DC solve: the driver role and its snapshot.

See also:
    docs/internals/primitive/xbar/solver.md
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from torch import Tensor


class ClampSnap(Protocol):
    """Required clamp-snapshot interface.

    Attributes:
        v_ref__V: NOMINAL reference or zero-current clamp voltage — the
            ideal value, carrying no driver-owned perturbation. A concrete
            snap keeps any offset / noise draw in its own dedicated
            field(s), folded in by that driver's own ``solve_clamp``.
    """

    @property
    def v_ref__V(self) -> Tensor: ...


SnapT = TypeVar("SnapT", bound=ClampSnap, contravariant=True)


class ClampDriver(Protocol[SnapT]):
    """What a boundary clamp exposes to one DC solve.

    A solve needs exactly one thing from a rail boundary: a transfer law it
    can evaluate at the port current inside its Newton loop, against a snap
    already sampled by the caller. That is the whole structural role, so any
    clamp circuit — a driver, a switched capacitor, a diode-connected load,
    a transimpedance amplifier held in clamp — satisfies it. Sampling the
    per-call snap and delivering the clamp at the converged state are each a
    concrete circuit's own method, named and shaped by that circuit and
    called by the owner that built it.
    """

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: SnapT,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Boundary clamp solve: returns ``(v_clamp__V, dVclamp_dI__MOhm)``.

        Args:
            i_port__uA: Port-output current.
            snap: Per-call snap, sampled by the concrete driver's own
                snapshot method.
            v_clamp_init__V: Optional warm-start hint.

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``.
        """
        ...
