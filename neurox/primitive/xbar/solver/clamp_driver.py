"""The rail boundary clamp of one DC solve: the driver role and its snapshot."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class ClampSnap(Protocol):
    """Required clamp-snapshot interface."""

    @property
    def v_open__V(self) -> Tensor:
        """Zero-load output voltage, including sampled driver non-idealities."""
        ...


class ClampDcop(Protocol):
    """Port voltage and local current derivative of a clamp operating point."""

    @property
    def v_port__V(self) -> Tensor:
        """Port voltage at the solved operating point."""
        ...

    @property
    def dvport_di__MOhm(self) -> Tensor:
        """Port-voltage derivative against the port current.

        Derivative-defined: a clamp whose slope is not a stored constant
        evaluates it at the operating point rather than reporting an
        output-resistance config field.
        """
        ...


class ClampDriver[SnapT: ClampSnap, DcopT: ClampDcop](Protocol):
    """What a boundary clamp exposes to one DC solve.

    The whole structural role is a transfer law the solve evaluates at the port
    current inside its Newton loop, against an already sampled snap. Sampling
    that snap and delivering the clamp at the converged state are each
    properties of the concrete circuit.

    Implementers must return local derivatives evaluated at the requested
    operating point with the same broadcast layout as the returned signal. A
    solver may invoke this transfer repeatedly on one held snapshot. Keep calls
    free of new physical sampling, programming, or energy billing; numerical
    iterations are not additional physical accesses.
    """

    def solve_dc(
        self,
        i_port__uA: Tensor,
        *,
        snap: SnapT,
        v_port_init__V: Tensor | None = None,
    ) -> DcopT:
        """Solve the driver's DC state through its port.

        Args:
            i_port__uA: Port-output current the solve enters through.
            snap: Per-call snap, sampled by the concrete driver's own snapshot
                method.
            v_port_init__V: Optional initial port voltage for a warm start.

        Returns:
            The module's own dcop, satisfying `ClampDcop`.
        """
        ...
