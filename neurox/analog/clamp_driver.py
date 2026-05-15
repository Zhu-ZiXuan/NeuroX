"""Port clamp-driver boundary protocol for the 1T1R array solver.

A 1T1R crossbar's BL **and** SL boundaries are not vanilla
post-stage readouts: each is a *non-ideal voltage source* whose
actual clamp level wobbles with the port current.  The solver's
outer Newton loop needs to know exactly two things about each
boundary at every iteration:

1. the actual clamp voltage produced by the driver at the present
   port current,
2. the local input impedance ``d v_clamp / d i_port`` so the wire
   Jacobian can fold the boundary's small-signal response in via a
   rank-1 Sherman-Morrison update.

The same Protocol applies to both axial boundaries — the solver
treats BL and SL as two applications of the same port clamp-driver
interface, with each port getting its own rank-1 fold-in on the
corresponding wire axis.

It does **not** need ``v_out``, internal device variables, or energy
counters — those belong to higher-level abstractions (readout, energy
accumulator) that live outside the solver.

The :class:`ClampDriver` Protocol expresses that minimum-viable
contract: a :attr:`v_ref__V` property for the Padé warm start, plus
:meth:`solve_clamp` returning the pair of tensors the BL Newton step
consumes.  Lifecycle methods that exist on every concrete driver —
``fabricate(...)`` for one-shot mismatch sampling and ``snapshot(...)``
for the per-VMM runtime token — are intentionally *not* part of this
Protocol; they are orchestrated by :class:`~neurox.xbar.Core1T1R`
using the concrete driver type (``OpAmpTIA``, ``Driver``, ...) where the
shape / return-type signatures naturally differ between
implementations.  The solver-facing surface stays narrow and stable.

There is intentionally **no** ``ClampDriverDC`` return-value Protocol
and **no** ``ClampDriverSnapshot`` snapshot Protocol.  Earlier
revisions tried both — return-value-as-Protocol and snapshot-as-
empty-marker-Protocol — and mypy treated method return / parameter
types structurally in ways that produced unstable diagnostics across
the concrete dataclass returns of OpAmpTIA / Driver.  The pragmatic fix is
to make the solver-facing types explicit at the surface:
:meth:`solve_clamp` returns the plain
``tuple[Tensor, Tensor]`` (``v_clamp__V``, ``dVclamp_dI__MOhm``) the
solver actually consumes, and accepts the per-VMM snapshot as an
opaque ``object`` token — concrete drivers narrow it back to their
own snapshot dataclass internally (``OpAmpTIASnapshot`` / ``DriverSnapshot``
/ ...) since the orchestration layer always pairs each driver with
its matching snapshot.

Concrete implementations live next to their physical models:

* :class:`~neurox.analog.driver.Driver` — ideal constant-voltage clamp
  driver.  Its :meth:`Driver.solve_clamp` returns the snapshot's noisy
  clamp voltage and zero input impedance; the richer
  :meth:`Driver.solve_dc` is still available for callers that want
  the full :class:`~neurox.analog.driver.DriverDC` result.
* :class:`~neurox.analog.opamp_tia.OpAmpTIA` — non-linear op-amp + pseudo-resistor
  feedback driver.  Its :meth:`OpAmpTIA.solve_clamp` is a thin wrapper
  around :meth:`OpAmpTIA.solve_dc`; higher layers that need ``v_out__V``
  call :meth:`OpAmpTIA.solve_dc` directly off the concrete instance.

Future implementations (``CSA``, ...) plug in by exposing the two
:class:`ClampDriver` members; their concrete ``solve_dc`` can return
any richer dataclass without the solver-facing Protocol caring.
"""

from typing import Protocol, runtime_checkable

from torch import Tensor

# ---------------------------------------------------------------------------
# Clamp-driver Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ClampDriver(Protocol):
    """Duck-typed port clamp driver consumed by the 1T1R solver.

    A concrete clamp driver qualifies if it exposes the
    :attr:`v_ref__V` property and the :meth:`solve_clamp` method.
    The solver's init takes two ``ClampDriver`` slots —
    ``bl_clamp_driver`` and ``sl_clamp_driver`` — and treats both
    opaquely.  It never calls ``fabricate(...)`` or ``snapshot(...)``
    itself, those are handled by the orchestration layer
    (:class:`~neurox.xbar.Core1T1R`) which holds the concrete driver
    types.

    Members:
        v_ref__V: Ideal / zero-current clamp voltage [V].  The solver
            reads this once per :meth:`solve` call to seed the
            Padé warm start without yet running the driver's
            :meth:`solve_clamp`.  For a OpAmpTIA it is the op-amp
            reference; for an ideal constant-voltage driver it is
            the nominal drive level; for a CSA it would be the
            steady-state nominal clamp at idle.
        solve_clamp: Solver-facing minimal interface.  Returns the
            two tensors the wire Newton step needs —
            ``(v_clamp__V, dVclamp_dI__MOhm)`` — at the present
            port current under the given snapshot.

            The first argument ``i_port__uA`` is the **port-output
            current** [uA]: positive when current leaves the clamp
            port (driver is *sourcing* into whatever circuit it
            terminates), negative when current enters the port
            (driver is *sinking*).  For a BL clamp, ``i_port`` is
            typically positive (the BL driver sources column
            current into the array); for an SL clamp it is
            typically negative (the SL driver sinks the cell
            return current to ground).  Concrete drivers compute
            this via Ohm's law on the first wire segment using the
            ``v_port − v_next_node`` convention so the sign stays
            consistent end-to-end — see :class:`Driver` and
            :class:`OpAmpTIA` for the two reference implementations.
            The name is deliberately port-agnostic: this Protocol
            applies to *any* port the driver terminates, BL or SL.

            The ``snapshot`` argument is typed as ``object`` because
            each concrete driver pairs its ``solve_clamp`` with its
            own snapshot dataclass (``OpAmpTIASnapshot``,
            ``DriverSnapshot``, ...); the Protocol intentionally
            does not constrain the snapshot type so the structural
            check stays focused on the two callables the solver
            actually invokes.  Concrete drivers typically implement
            this as a thin extraction on top of their own richer
            ``solve_dc(...)`` so the physical solve runs once.  The
            tuple ordering is fixed: entry 0 is the clamp voltage,
            entry 1 is the boundary input impedance.

    Note:
        Concrete drivers (OpAmpTIA, Driver) additionally expose
        ``fabricate(shape)``, ``snapshot(shape=...)``, and their own
        ``solve_dc(...)`` returning a circuit-specific dataclass
        (``OpAmpTIADC`` / ``DriverDC`` / etc.).  Those richer surfaces are
        **not** part of this Protocol — they are reserved for callers
        outside the solver (Core1T1R orchestration, readout, energy
        aggregation) that need access to lifecycle hooks or
        circuit-specific quantities such as the OpAmpTIA's ``v_out__V``.
        The Protocol is intentionally narrow.
    """

    @property
    def v_ref__V(self) -> float: ...

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snapshot: object,
        *,
        v_clamp_init__V: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...
