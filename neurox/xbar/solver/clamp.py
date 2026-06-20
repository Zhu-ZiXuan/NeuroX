"""Structural clamp-driver role consumed by the array solver.

The array solver drives each boundary port through a clamp circuit and
needs only three capabilities from it: a reference clamp voltage, a
per-call snap of fabricated state, and a clamp solve mapping port current
to ``(v_clamp, dVclamp/dI)``. :class:`ClampDriver` names that capability
contract as a structural (``Protocol``) role rather than a registry base
class: there is no inheritance and no ``RegistryMixin``. The role lives
beside the solver because the solver is its only consumer; concrete
clamps (the ``TIA`` family, the ideal ``Driver``) live in
``neurox/analog`` and satisfy it structurally, without importing it.

The role is generic over ``SnapT`` so that each conforming circuit ties
its own :meth:`ClampDriver.snapshot` output to its
:meth:`ClampDriver.solve_clamp` input, keeping the snap type consistent
end to end without forcing a shared snap hierarchy. ``SnapT`` is
deliberately unbounded: conforming snaps need not share a base, so the
TIA family (e.g. ``OpAmpTIA`` as ``ClampDriver[OpAmpTIASnap]``), the
ideal ``Driver`` (``ClampDriver[DriverSnap]``), and a future CSA each
satisfy the role structurally without declaring inheritance.

See also:
    docs/internals/xbar/solver.md
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from torch import Tensor

SnapT = TypeVar("SnapT")


class ClampDriver(Protocol[SnapT]):
    """Structural contract any boundary clamp circuit satisfies.

    Attributes:
        v_ref__V: Reference / zero-current clamp voltage [V].

    Methods:
        snapshot: Sample one per-call snap of the fabricated state over a
            broadcast ``shape``, optionally selecting a chunk via
            ``multi_coords``.
        solve_clamp: Boundary clamp solve mapping port current to the
            clamp voltage and its small-signal slope.
    """

    @property
    def v_ref__V(self) -> float:
        """Reference / zero-current clamp voltage [V]."""
        ...

    def snapshot(self, *, shape: tuple[int, ...], multi_coords: tuple[Tensor, ...] | None) -> SnapT:
        """Sample one per-call runtime snap over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snap fills tensor fields
                at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state.
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
            i_port__uA: Port-output current [uA].
            snap: Per-call snap from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start hint [V].

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``.
        """
        ...
