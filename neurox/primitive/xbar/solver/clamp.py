"""The rail boundary clamp of one DC solve: the driver role and its snapshot.

See also:
    docs/internals/primitive/xbar/solver/clamp.md
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from torch import Tensor


class ClampSnap(Protocol):
    """Required clamp-snapshot interface."""

    @property
    def v_ref__V(self) -> Tensor:
        """NOMINAL reference or zero-current clamp voltage, carrying no
        driver-owned perturbation. A concrete snap keeps any offset or noise
        draw in its own dedicated field(s), folded in by `solve_clamp`."""
        ...


SnapT = TypeVar("SnapT", bound=ClampSnap, contravariant=True)


class ClampDriver(Protocol[SnapT]):
    """What a boundary clamp exposes to one DC solve.

    The whole structural role is a transfer law the solve evaluates at the
    port current inside its Newton loop, against a snap already sampled by the
    caller. Sampling that snap and delivering the clamp at the converged state
    are each a concrete circuit's own method.
    """

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: SnapT,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the boundary clamp at one port current.

        Args:
            i_port__uA: Port-output current.
            snap: Per-call snap, sampled by the concrete driver's own
                snapshot method.
            v_clamp_init__V: Optional warm-start hint.

        Returns:
            `(v_clamp__V, dVclamp_dI__MOhm)` — the clamp voltage and its
            slope against the port current.
        """
        ...
