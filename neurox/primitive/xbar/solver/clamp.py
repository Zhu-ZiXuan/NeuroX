"""Structural interface for boundary clamp drivers.

See also:
    docs/internals/primitive/xbar/solver.md
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from torch import Tensor


class ClampSnap(Protocol):
    """Required clamp-snapshot interface.

    Attributes:
        v_ref__V: Reference or zero-current clamp voltage.
    """

    @property
    def v_ref__V(self) -> Tensor: ...


SnapT = TypeVar("SnapT", bound=ClampSnap)


class ClampDriver(Protocol[SnapT]):
    """Structural contract any boundary clamp circuit satisfies."""

    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> SnapT:
        """Sample one per-call runtime snap over ``shape``.

        Args:
            v_ref__V: Injected reference / zero-current clamp voltage;
                the source-agnostic tap value carried into the snap, which
                the driver may perturb with its per-call nonidealities.
            shape: Per-call broadcast shape; the snap fills tensor fields
                at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state, carrying the reference
            ``v_ref__V``.
        """
        ...

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
            snap: Per-call snap from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start hint.

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``.
        """
        ...
