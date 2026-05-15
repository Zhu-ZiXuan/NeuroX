"""Fully-flat DC solver for a 1T1R crossbar array with paired clamp drivers.

Each cell is a series stack of an RRAM and an NMOS access transistor
with an intermediate node ``v_x_node``:

    v_bl_node ── RRAM ── v_x_node ── NMOS ── v_sl_node

Both axial boundaries are owned by the solver: a **BL clamp driver**
(today either the non-linear :class:`~neurox.analog.opamp_tia.OpAmpTIA` for a
closed-loop op-amp + pseudo-resistor cascade or the ideal
:class:`~neurox.analog.driver.Driver` for a constant-voltage clamp)
and a **SL clamp driver** (today the ideal :class:`Driver` — the
future non-ideal source-side clamp drops in here without touching the
solver).  Both consume the :class:`~neurox.analog.ClampDriver`
Protocol's :meth:`solve_clamp`, which returns the plain
``tuple[Tensor, Tensor]`` ``(v_clamp__V, dVclamp_dI__MOhm)`` the
wire-Newton step needs.  Concrete drivers still own a richer
``solve_dc(...)`` returning a circuit-specific dataclass (``OpAmpTIADC``,
``DriverDC``, ...); the structural Protocol is intentionally narrow
so the solver never has to reason about those concrete result types,
and future drivers (``CSA``, ...) plug in by adding their own
``solve_clamp`` wrapper on top of whatever physical solve they need.

Naming shorthand: inside this solver "driver" abbreviates
"clamp_driver" — slot attributes ``self.bl_driver`` /
``self.sl_driver``, snapshot locals ``bl_driver_snapshot`` /
``sl_driver_snapshot``, port-current locals ``i_*_driver_in__uA``,
boundary impedance locals ``r_*_driver_in__MOhm``.  Boundary
voltages keep the physical short names ``v_bl_clamp`` /
``v_sl_drive`` — they refer to the voltage *at* the boundary, not
to the driver object.

Each driver's small-signal input impedance ``dVclamp_dI__MOhm`` is
folded directly into its wire's Jacobian via a rank-1 Sherman-Morrison
update.  Every outer iteration first solves the per-cell KCL, then
asks each driver for its new ``(v_clamp, r_in)`` at the latest
**driver port current** computed from the boundary KCL — the same
``delta_v · g`` form :func:`col_driver_current` /
:func:`row_driver_current` returns at convergence.  For an ideal
driver ``r_in__MOhm == 0`` and the rank-1 correction vanishes, so
the wire-Newton step degenerates to the plain tridiagonal solve
without any conditional in the kernel.  Because the BL and SL
rank-1 fold-ins share the same Sherman-Morrison structure (only the
wire axis and the wire's driver-segment conductance differ), they
are implemented by a single shared helper
:meth:`_solve_driver_newton_rank1` parametrised by the active
``dim`` and the driver-segment scalar.

Clamp-driver port current
-------------------------
The value passed into :meth:`ClampDriver.solve_clamp` represents
the **net current at the clamp-driver port** — see
``temp/sl_driver_solver.md`` for the design discussion.  The
Protocol parameter is named ``i_port__uA`` and follows the
port-output convention (positive when current leaves the clamp into
the external circuit); inside the solver the locals that hold those
quantities are ``i_bl_driver_in__uA`` / ``i_sl_driver_in__uA``.

The solver computes each port current from the **boundary KCL** —
the Ohmic flow across the driver-to-first wire segment:

* BL: ``i_bl_driver_in__uA = (v_bl_clamp - v_bl_node[..., 0]) ·
  bl_segment_g[0]``.
* SL: ``i_sl_driver_in__uA = (v_sl_drive - v_sl_node[..., 0, :]) ·
  sl_segment_g[0]``.

This keeps the clamp-driver interface topology-agnostic: the
cell-side identities ``i_bl_driver_in__uA = i_cell.sum(dim=-1)`` and
``i_sl_driver_in__uA = -i_cell.sum(dim=-2)`` only hold on the
current 1-D BL / SL ladders (no branch paths, no leakage, open far
end) and would silently break under any richer wire topology; the
boundary KCL form generalises without changing the solver contract.
At convergence both forms agree; only the boundary KCL form is used
inside the outer Newton iteration.  The very first warm-up call —
before any IR-drop wire voltages have been computed — uses the
cell-side sum since no consistent boundary state exists yet.

Naming conventions
------------------
Two strictly separated families:

A. **owner-style** ``<owner>_<property>`` — used **only** for
   intrinsic properties of an entity (device, component, structural
   element).  Examples: ``bl_wire.segment_r__MOhm`` (BL wire's
   per-segment resistance tensor), ``nmos_beta__uA_per_V2``
   (NMOS's β), ``rram_state_g`` (RRAM's state-conductance
   snapshot).  These are configuration / fabrication snapshots,
   never solver-state variables.

B. **quantity-first** ``<type>_<subscript>`` — used for solver
   state, node voltages, boundary voltages, residuals, increments,
   derivatives.  The leading symbol carries the physical quantity
   (``v_*`` voltage, ``i_*`` current, ``g_*`` conductance,
   ``f_*`` residual, ``dv_*`` Newton increment), and the rest is a
   mathematical subscript — *not* an owner.  Examples:
   ``v_bl_node`` = ``v_{bl,node}``, ``g_cell_eff`` = ``g_{cell,eff}``,
   ``r_bl_driver_in__MOhm`` / ``r_sl_driver_in__MOhm`` = boundary
   input impedances reported by the two clamp drivers.

The two families never mix per name.  Unit suffixes (``__V``,
``__uA``, ``__uS``, ``__MOhm``) appear on boundary-crossing
identifiers (solve()'s kwargs, registered buffers) and on the
device APIs' returned dataclass fields; they may be dropped on
purely internal solver locals.

Solver lifecycle
----------------
:class:`NewtonRaphsonSolver1T1R` is an ``nn.Module`` bound only to
the *structural* per-tile inputs at construction: the ``rram`` /
``nmos`` / ``bl_driver`` / ``sl_driver`` per-instance modules and
the two fabricated :class:`~neurox.device.Wire` instances
(``bl_wire`` / ``sl_wire``).  Array geometry is **not** baked into
the solver — ``num_col`` / ``num_row`` are read fresh from
``rram_snapshot.state_g__uS.shape`` on every :meth:`solve` call,
and per-segment wire resistance / conductance tensors are read
from the wire modules' own state buffers; the tridiagonal Jacobian
templates are derived locally from those segment tensors (see
``temp/wire.md``).  An :class:`Offset1T1RXbar` instantiates one
solver per :class:`Core1T1R` and reuses it across every VMM.

Per-VMM device-specific state lives inside the module instances
themselves — :class:`~neurox.device.RRAM` owns its post-program
``state_g__uS``, :class:`~neurox.device.NMOS` owns its fabricated
``β`` / ``V_th``, the OpAmpTIA owns its per-column ``opamp_gain`` plus
its internal NMOS submodule's mismatch buffers, and each clamp
driver owns its snapshot's noisy clamp voltage.  The solver takes
four per-VMM **runtime-state values** sampled once per
:meth:`solve` call by the caller and threads them through every
internal helper — the RRAM and NMOS snapshots are typed concretely
(:class:`~neurox.device.RRAMSnapshot`,
:class:`~neurox.device.NMOSSnapshot`) since those modules own
fixed dataclass shapes; the two driver snapshots are typed as the
bare ``object`` matching :meth:`ClampDriver.solve_clamp`'s
opaque-token discipline (each concrete driver pairs ``solve_clamp``
with its own snapshot type without leaking that type through the
solver's Protocol surface).  No per-call attribute stashing on
``self`` — ``@torch.compile``'s Dynamo tracer treats attribute
writes on a Python object as opaque side effects, so any helper
that read them via ``self.xxx`` would either inhibit fusion or
force a graph break.

Single solve path — **free of dynamic Python control flow**
-----------------------------------------------------------
A single compiled kernel performs, in fused order:

    a. **Warm start phase 1** — Padé / closed-form V_X solve at
       ``v_bl_node = bl_driver.v_ref__V`` and
       ``v_sl_node = sl_driver.v_ref__V`` gives ``i_cell_init``
       and ``v_x_node_init`` in **no inner Newton loop**.
    b. **First clamp-driver evaluations** — sum ``i_cell_init`` along
       both axes, ask each driver for its steady-state
       ``(v_clamp__V, r_in__MOhm)``.  This seeds the IR-drop warm
       start with physically-consistent boundaries instead of
       hard-coded reference voltages.
    c. **Warm start phase 2** — first-order IR-drop wire voltages
       derived from ``i_cell_init`` and the two warm-start clamp
       voltages.
    d. ``N_UNROLL_OUTER`` fully unrolled outer Newton iterations on
       ``(V_BL, V_SL)``.  Inside each iteration: incremental cell
       Newton seeded from the previous ``v_x_node``, both
       boundary-KCL clamp-port currents + clamp-driver calls →
       new ``v_bl_clamp`` / ``v_sl_drive`` + ``r_in``, KCL
       residuals, and BL / SL wire corrections that fold each
       driver's ``r_in`` into the corresponding Jacobian as a
       rank-1 update (Sherman-Morrison on top of the Thomas
       tridiagonal solve) — both axes go through the shared
       :meth:`_solve_driver_newton_rank1` helper.  No Python loop,
       no ``.item()`` sync.
    e. Driver-port currents extracted from the final ``v_bl_node`` /
       ``v_sl_node`` and the converged clamp voltages.

Definition domain
-----------------
The wire model is a strictly-positive finite **1-D resistor ladder**:
every entry of ``bl_wire.segment_r__MOhm`` /
``sl_wire.segment_r__MOhm`` **must be > 0**.
:meth:`~neurox.device.Wire.fabricate` enforces this on the wire
side.  Ideal-wire / zero-resistance topologies are out of scope and
belong to a separate ideal-xbar path; mixing zero and non-zero wire
segments is not physically defined for this solver and is not
supported.  Source-line topology is implicitly ``"row_shared"`` —
the SL extends along the row axis, just like the WL; column-shared
SL layouts (``sl_topology="col_shared"``) are reserved for a future
build and are rejected up at :meth:`Core1T1R.fabricate` time (see
``temp/wire.md``).  Future strap / bridge / comb source-line
networks will not extend this solver — they would live in a
separate general-network solve since the current Jacobian relies on
the tridiagonal-plus-rank-1 structure of a pure 1-D ladder.

With the warm start in the quadratic-convergence basin and the
rank-1 boundary fold-ins giving a proper coupled Newton step on
``(V_BL, V_blClamp, V_SL, V_slDrive)``, ``N_UNROLL_OUTER = 5`` gives
a comfortable safety margin.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.analog.clamp_driver import ClampDriver
from neurox.device import RRAM, RRAMSnapshot, Wire
from neurox.device.nmos import NMOS, NMOSSnapshot
from neurox.xbar.solver import (
    col_driver_current,
    col_wire_kcl_residual,
    row_driver_current,
    row_wire_kcl_residual,
    solve_tridiagonal,
)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverResult:
    """Complete steady-state solution of a 1T1R xbar DC solve.

    Every field is a torch Tensor computed inside a single compiled
    solver kernel.  A frozen dataclass is used instead of a
    NamedTuple because ``torch.compile`` traces frozen dataclasses
    cleanly while NamedTuple subclasses hit a known dynamo
    ``IndexError`` on tagged tuples.

    All field names are **quantity-first** (mathematical-subscript
    style): the leading symbol is the physical quantity (``v_``,
    ``i_``), and the suffix is the position / role.  ``i_cell``
    keeps its standard solver-state form.

    Attributes:
        i_bl_driver: Net current from each BL driver into the wire
            [uA].  Sign convention: driver-into-wire (positive when
            the BL clamp is sourcing current into the array).
            Shape: ``[*batch, col_num]``.
        i_sl_driver: Net current from each SL driver into the wire
            [uA].  Sign convention: driver-into-wire (negative when
            the SL clamp absorbs cell current to ground).
            Shape: ``[*batch, row_num]``.
        v_bl_node: Steady-state BL node voltage at every (col, row)
            position [V].  In the ideal-wire limit this is a
            broadcast view of the dynamic ``v_bl_clamp``; with
            finite wire resistance it varies along the column due
            to IR drop.  Shape: ``[*batch, col_num, row_num]``.
        v_sl_node: Steady-state SL node voltage at every (col, row)
            position [V].  Broadcast of the SL clamp voltage for
            ideal wires; varies along the row for finite wires.
            Shape: ``[*batch, col_num, row_num]``.
        v_x_node: Steady-state Node-X voltage at every cell [V] —
            the intermediate node between the RRAM bottom electrode
            and the NMOS drain.  Always a per-cell tensor.
            Shape: ``[*batch, col_num, row_num]``.
        i_cell: Steady-state per-cell current [uA], computed as
            ``g_sw · (V_X − V_SL)`` at the converged voltages.
            Shape: ``[*batch, col_num, row_num]``.
        v_bl_clamp: Dynamic per-column BL clamp voltage [V] reported
            by the BL clamp driver at the converged operating point.
            For a finite-gain driver (OpAmpTIA) it droops below the
            ideal ``v_ref`` as the column current rises.  Shape:
            ``[*batch, col_num]``.
        v_sl_drive: Dynamic per-row SL drive voltage [V] reported by
            the SL clamp driver at the converged operating point.
            For an ideal :class:`~neurox.analog.driver.Driver` this
            is the snapshot's noisy nominal drive voltage; a future
            non-ideal source-side driver will shift dynamically
            with the row's net cell current.  Shape:
            ``[*batch, row_num]``.

    Note:
        The OpAmpTIA-specific ``v_out`` field is not on this solver-
        facing result.  The readout path consumes ``v_out`` by
        re-invoking ``self.tia.solve_dc(solver_result.i_bl_driver,
        bl_driver_snapshot, v_clamp_init__V=solver_result.v_bl_clamp)``
        in :meth:`Core1T1R.forward` after the solver returns — the
        returned :class:`~neurox.analog.opamp_tia.OpAmpTIADC` carries
        ``v_out__V``.  ``i_bl_driver`` is the converged clamp-port
        current (the same boundary-KCL ``delta_v · g`` quantity the
        solver fed into every ``solve_clamp`` call), so the
        post-solver re-evaluation operates on a port current
        consistent with the solved boundary.  Per
        ``temp/clamp_driver.md``, the solver-facing contract carries
        only the boundary working point and input impedance.
    """

    i_bl_driver: Tensor
    i_sl_driver: Tensor
    v_bl_node: Tensor
    v_sl_node: Tensor
    v_x_node: Tensor
    i_cell: Tensor
    v_bl_clamp: Tensor
    v_sl_drive: Tensor


class NewtonRaphsonSolver1T1R(nn.Module):
    """Stateful 1T1R DC solver coupled to two clamp drivers (BL + SL).

    Holds the shared, per-tile structural inputs at construction:

    * ``rram`` / ``nmos`` — stateless array device tools (registered
      as submodules so ``.to(device)`` propagates their internal
      buffers).
    * ``bl_driver`` / ``sl_driver`` — two objects satisfying the
      :class:`~neurox.analog.ClampDriver` Protocol (the
      ``*_driver`` slot names abbreviate "BL clamp_driver" / "SL
      clamp_driver" per the module-level naming shorthand).  Today
      the BL slot is :class:`~neurox.analog.opamp_tia.OpAmpTIA` (closed-loop
      op-amp + pseudo-resistor) or the ideal
      :class:`~neurox.analog.driver.Driver`; the SL slot is the
      ideal :class:`Driver`.  Future CSA-class boundaries (either
      side) satisfy the same Protocol.  The solver calls each
      driver only through
      :meth:`~neurox.analog.ClampDriver.solve_clamp`, which returns
      the plain ``(v_clamp__V, dVclamp_dI__MOhm)`` tuple the wire
      Newton step needs — circuit-specific dataclasses (``OpAmpTIADC``,
      ``DriverDC``) live on the concrete classes and are reached by
      higher layers via their own ``solve_dc(...)`` entry points.
    * ``bl_wire`` / ``sl_wire`` — the two fabricated
      :class:`~neurox.device.Wire` instances that own the per-segment
      resistance / conductance tensors.  No tile geometry is stored
      on ``self``: the wire-Newton helpers read
      ``bl_wire.segment_g__uS`` / ``bl_wire.segment_r__MOhm`` (and
      SL equivalents) directly per call and assemble their
      tridiagonal Jacobian templates locally (per
      ``temp/wire.md``).

    Per-VMM runtime-state values are passed to :meth:`solve` as
    plain arguments and threaded through every internal helper the
    same way — :class:`~neurox.device.RRAMSnapshot` and
    :class:`~neurox.device.NMOSSnapshot` are concrete dataclasses
    (their owning modules expose fixed snapshot shapes), while the
    two driver snapshots are typed as ``object`` to match
    :meth:`~neurox.analog.ClampDriver.solve_clamp`'s opaque-token
    Protocol — each concrete driver class narrows it back to its
    own snapshot dataclass internally.  No per-call attribute
    stashing on ``self`` — ``@torch.compile``'s Dynamo tracer
    treats attribute writes on a Python object as opaque side
    effects, so any helper that read them via ``self.xxx`` would
    either inhibit fusion or force a graph break.

    Class attributes:
        N_UNROLL_OUTER: Fixed depth of the fully-unrolled outer
            Newton loop.  With ``r_in__MOhm`` folded into both wire
            Jacobians via Sherman-Morrison, ``5`` accommodates
            residual non-linearity in the cell + wire couplings at
            heavy column draw.
        I_ATOL__uA: Absolute per-node KCL residual tolerance in uA
            (informational; the flat kernel does not check against
            this at runtime — convergence is ensured by running
            ``N_UNROLL_OUTER`` iterations from a warm-started state).

    Args:
        rram: Stateless RRAM tool.
        nmos: Stateless array NMOS tool (per-cell access transistor).
        bl_driver: BL clamp driver satisfying the
            :class:`~neurox.analog.ClampDriver` Protocol.  Concrete
            types today are :class:`~neurox.analog.opamp_tia.OpAmpTIA` (closed-
            loop) and :class:`~neurox.analog.driver.Driver` (ideal
            constant-voltage); a future CSA implementation can plug
            in here with no changes to the solver.
        sl_driver: SL clamp driver satisfying the
            :class:`~neurox.analog.ClampDriver` Protocol.  Today
            :class:`~neurox.analog.driver.Driver` (ideal
            constant-voltage); a future non-ideal source-side
            driver plugs in without touching the solver.
        bl_wire: Fabricated BL :class:`~neurox.device.Wire` instance.
            The solver reads its ``segment_g__uS`` /
            ``segment_r__MOhm`` buffers; ideal-wire / zero-resistance
            topologies are out of scope and must go through a
            separate ideal-xbar path.
        sl_wire: Fabricated SL :class:`~neurox.device.Wire` instance.
    """

    N_UNROLL_OUTER: int = 5
    I_ATOL__uA: float = 1e-3

    def __init__(
        self,
        rram: RRAM,
        nmos: NMOS,
        bl_driver: ClampDriver,
        sl_driver: ClampDriver,
        *,
        bl_wire: Wire,
        sl_wire: Wire,
    ) -> None:
        super().__init__()
        self.rram = rram
        self.nmos = nmos
        self.bl_driver = bl_driver
        self.sl_driver = sl_driver
        self.bl_wire = bl_wire
        self.sl_wire = sl_wire

    # ---------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------

    def solve(
        self,
        v_wl_drive__V: Tensor,
        *,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
        bl_driver_snapshot: object,
        sl_driver_snapshot: object,
    ) -> SolverResult:
        """Run the Padé-warm-start + coupled Newton solver.

        Per-VMM module state is supplied entirely through the four
        runtime-state dataclasses; nothing is stashed on ``self``
        between calls.  The runtime snapshots are **fixed for the
        whole solve** — every internal helper reads from the same
        objects so the Newton iteration sees a consistent set of
        noisy values (per ``temp/state_holding.md``).

        Wire state is **not** an argument: it is read directly from
        ``self.bl_wire`` / ``self.sl_wire`` (the fabricated
        :class:`~neurox.device.Wire` instances).  This shifts every
        line-geometry concern into the wire module and frees the
        solver from per-call template construction (see
        ``temp/wire.md``).

        Two RRAM conductance tensors are used inside this method, and
        the distinction is load-bearing (see ``temp/1t1r_solver.md``):

        * ``rram_state_g_static__uS`` — the **static programmed**
          per-cell conductance read directly from
          ``self.rram.state_g__uS``.  Used **only** for the
          divider-style warm-start initialisation, where a stable
          first-order seed is preferred to the noisy snapshot.
        * ``rram_snapshot.state_g__uS`` — the **noisy per-VMM**
          read-noise snapshot.  Drives every real non-linear cell
          evaluation inside the Newton iteration; threaded through
          ``RRAM.solve_dc(...)`` calls so the converged answer
          reflects the runtime noise sample.

        Args:
            v_wl_drive__V: Per-cell NMOS gate (WL) voltage [V].  In
                the 1T1R xbar this is ``wl_logic · V_DD,WL``
                broadcast across the column axis.  Shape: any
                tensor broadcastable to ``[*batch, num_col, num_row]``
                (typically ``[*batch, 1, num_row]``).  Kept as an
                explicit external argument because the WL DAC is not
                modelled as a coupled boundary solve.
            rram_snapshot: RRAM per-VMM read-noise snapshot from
                :meth:`RRAM.snapshot`.  Drives the shape of the
                solve: ``rram_snapshot.state_g__uS.shape`` is the
                authoritative ``[*batch, num_col, num_row]`` layout.
            nmos_snapshot: Array NMOS per-VMM dynamic snapshot from
                :meth:`NMOS.snapshot` (empty today).
            bl_driver_snapshot: Per-VMM BL clamp-driver dynamic
                snapshot, typed as the opaque ``object`` token that
                :meth:`~neurox.analog.ClampDriver.solve_clamp`
                accepts.  Sampled by the caller off the concrete BL
                driver instance (``OpAmpTIA.snapshot(...)`` returns a
                :class:`~neurox.analog.opamp_tia.OpAmpTIASnapshot`;
                ``Driver.snapshot(...)`` returns a
                :class:`~neurox.analog.driver.DriverSnapshot`).
            sl_driver_snapshot: Per-VMM SL clamp-driver dynamic
                snapshot, typed identically to
                ``bl_driver_snapshot`` but produced off
                ``self.sl_driver``.

        Returns:
            :class:`SolverResult` with the converged driver-port
            currents, node voltages, per-cell current, and the
            BL / SL clamp voltages.
        """
        # Per-segment wire state — owned by the Wire modules.
        bl_segment_r = self.bl_wire.segment_r__MOhm
        sl_segment_r = self.sl_wire.segment_r__MOhm
        bl_segment_g = self.bl_wire.segment_g__uS
        sl_segment_g = self.sl_wire.segment_g__uS

        # Tridiagonal-Jacobian wire templates — derived once per solve
        # from the static segment conductances and threaded into every
        # outer Newton iteration.  They are invariant across the
        # ``N_UNROLL_OUTER`` iterations (only the cell-Newton state and
        # the two clamp boundaries change), so computing them here
        # avoids ``N_UNROLL_OUTER`` redundant pad / add ops inside the
        # wire-Newton helpers.
        #
        #   wire_diag[k]    = segment_g[k] + segment_g[k+1]   (k <= N-2)
        #   wire_diag[N-1]  = segment_g[N-1]                  (open far end)
        #   wire_offdiag    = -segment_g[1:]                  (length N-1)
        bl_wire_diag = bl_segment_g + F.pad(bl_segment_g[1:], (0, 1))
        bl_wire_offdiag = -bl_segment_g[1:]
        sl_wire_diag = sl_segment_g + F.pad(sl_segment_g[1:], (0, 1))
        sl_wire_offdiag = -sl_segment_g[1:]
        # Driver-edge conductance scalars threaded into the shared
        # rank-1 helper.  Reading them off the wire here keeps the
        # helper module-reference-free.
        bl_driver_segment_g__uS = bl_segment_g[0]
        sl_driver_segment_g__uS = sl_segment_g[0]

        # Execution-shape source — authoritative layout for the solve.
        # The noisy per-VMM snapshot drives every real cell evaluation
        # inside the Newton loop.
        rram_state_g_snapshot = rram_snapshot.state_g__uS
        *batch, num_col, num_row = rram_state_g_snapshot.shape
        # Static programmed conductance — expanded once to the
        # execution shape so the warm-start divider sees a clean
        # broadcast.  Per ``temp/1t1r_solver.md`` this is intentionally
        # the *non-noisy* path: the warm start only needs a
        # first-order operating point, and using the static buffer
        # keeps initialisation decoupled from the runtime snapshot
        # representation.
        rram_state_g_static = self.rram.state_g__uS.expand_as(rram_state_g_snapshot)
        # Solver definition domain: at least two nodes on each wire.
        # The KCL residual uses ``torch.diff`` along both axes, and the
        # wire-conductance diagonal assumes a driver-segment plus at
        # least one inter-node segment.  ``XbarConfig`` already
        # validates this for any tile built through the config layer;
        # this guard catches direct callers that bypass the config.
        if not (num_col > 1):
            raise ValueError(f"require: num_col ({num_col}) > 1")
        if not (num_row > 1):
            raise ValueError(f"require: num_row ({num_row}) > 1")
        v_wl_drive_grid__V = v_wl_drive__V.expand(*batch, num_col, num_row)

        # --- Warm start phase 1: Padé i_cell at v_bl_node = v_bl_ref, v_sl_node = v_sl_ref ---
        # Each driver's ``v_ref__V`` (a property on the Protocol) is
        # its ideal / zero-current clamp voltage.  Using both
        # directly — rather than running extra zero-current
        # ``solve_dc`` calls — keeps the seed cheap and
        # driver-agnostic; the first real clamp-driver calls below
        # converge quickly from this static op.
        v_bl_clamp_ref__V = self.bl_driver.v_ref__V
        v_sl_drive_ref__V = self.sl_driver.v_ref__V
        v_bl_node_seed = torch.full_like(rram_state_g_snapshot, v_bl_clamp_ref__V)
        v_sl_node_seed = torch.full_like(rram_state_g_snapshot, v_sl_drive_ref__V)
        i_cell_init, _, v_x_node_init = self._solve_cell_pade_warm_start(
            v_bl_node_seed,
            v_sl_node_seed,
            v_wl_drive_grid__V,
            rram_state_g_static,
            rram_snapshot,
            nmos_snapshot,
        )

        # --- First clamp-driver evaluations (cold start from static op) ---
        # The :class:`~neurox.analog.ClampDriver` Protocol's
        # :meth:`solve_clamp` returns the two tensors the solver needs
        # as a plain ``tuple[Tensor, Tensor]``.  Concrete drivers
        # (OpAmpTIA's :class:`OpAmpTIADC`, Driver's :class:`DriverDC`) carry
        # richer return objects for higher-layer consumers, but those
        # are reached through each driver's own ``solve_dc(...)`` and
        # are intentionally not part of the solver-facing surface.
        #
        # Special case for these **warm-up calls only**: each clamp
        # port current is seeded with the cell-side sum.  We do not
        # yet have a converged boundary state — the IR-drop wire
        # voltages below depend on the warm-start clamp voltages —
        # so the boundary-KCL form would be evaluated against an
        # inconsistent operating point.  The cell-side sums are the
        # natural pre-clamp estimates and are exact for the 1-D
        # ladders by KCL conservation; subsequent outer Newton
        # iterations switch to the boundary-KCL form below as soon
        # as ``v_bl_clamp`` / ``v_sl_drive`` are known.
        #
        # Sign conventions:
        #   * BL: cells *draw* current from BL → port-output current
        #     is positive (driver sourcing into the wire).
        #     i_bl_port = + Σ_row i_cell.
        #   * SL: cells *source* current into SL → port-output
        #     current is negative (driver sinking from the wire).
        #     i_sl_port = − Σ_col i_cell.
        # Shape: i_bl_driver_in__uA -> [..., num_col]
        i_bl_driver_in__uA = i_cell_init.sum(dim=-1)
        # Shape: i_sl_driver_in__uA -> [..., num_row]
        i_sl_driver_in__uA = -i_cell_init.sum(dim=-2)
        v_bl_clamp__V, _ = self.bl_driver.solve_clamp(i_bl_driver_in__uA, bl_driver_snapshot)
        v_sl_drive__V, _ = self.sl_driver.solve_clamp(i_sl_driver_in__uA, sl_driver_snapshot)
        v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
        v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-2)

        # --- Warm start phase 2: first-order IR-drop wire voltages ---
        # BL wire runs along dim=-1, driver at index 0, cell currents +i_cell.
        # ``i_bl_downstream[k]`` is the current flowing through segment ``k``
        # (driver → first segment at k=0, inter-node segments for k>=1)
        # toward the open far end — equal to the sum of injected currents
        # at nodes ``k..N-1``.  Voltage drop across segment ``m`` is
        # ``i_bl_downstream[m] * bl_segment_r[m]``.  We seed
        # ``v_bl_node`` from the warm-start ``v_bl_clamp`` (just
        # computed) so the next outer iteration sees a boundary-KCL
        # port current consistent with the IR-drop seed.
        bl_shape_broadcast = [1] * rram_state_g_snapshot.ndim
        bl_shape_broadcast[-1] = num_row
        bl_segment_r_broadcast = bl_segment_r.view(bl_shape_broadcast)
        i_bl_downstream = torch.flip(torch.cumsum(torch.flip(i_cell_init, [-1]), -1), [-1])
        v_bl_node = v_bl_clamp_grid__V - torch.cumsum(i_bl_downstream * bl_segment_r_broadcast, dim=-1)

        # SL wire runs along dim=-2, driver at index 0, cell currents
        # ``i_inject = -i_cell`` (cell sources +i_cell into SL).
        # Mirror the BL seed against the warm-start ``v_sl_drive``.
        sl_shape_broadcast = [1] * rram_state_g_snapshot.ndim
        sl_shape_broadcast[-2] = num_col
        sl_segment_r_broadcast = sl_segment_r.view(sl_shape_broadcast)
        i_sl_inject = -i_cell_init
        i_sl_downstream = torch.flip(torch.cumsum(torch.flip(i_sl_inject, [-2]), -2), [-2])
        v_sl_node = v_sl_drive_grid__V - torch.cumsum(i_sl_downstream * sl_segment_r_broadcast, dim=-2)

        # --- Outer Newton (fully unrolled, undamped) ---
        # Seed the cell Newton with the warm-start ``v_x_node_init``;
        # subsequent iterations reuse the previous iteration's
        # ``v_x_node`` (slowly-varying as ``v_bl_node`` /
        # ``v_sl_node`` move incrementally).
        v_x_node = v_x_node_init
        i_cell = i_cell_init
        g_cell_eff = torch.zeros_like(v_x_node)  # placeholder overwritten below
        for _ in range(self.N_UNROLL_OUTER):
            i_cell, g_cell_eff, v_x_node = self._solve_cell_newton_warm_start(
                v_bl_node,
                v_sl_node,
                v_wl_drive_grid__V,
                v_x_node,
                rram_snapshot,
                nmos_snapshot,
            )

            # Clamp-driver refreshes: ask each driver for the current
            # operating point + boundary input impedance at the latest
            # port current.  The clamp port currents are computed from
            # the **boundary KCL** — the Ohmic flow across each
            # driver-to-first wire segment using the previous outer
            # iteration's ``v_bl_clamp`` / ``v_sl_drive`` against the
            # current ``v_bl_node`` / ``v_sl_node``.  This is the
            # physically-fundamental quantity at each port and stays
            # correct under topologies the cell-sum identity would
            # silently break.  ``v_clamp_init__V`` is forwarded to
            # each driver's internal Newton (OpAmpTIA contracts from a
            # near-converged seed; ideal :class:`Driver` ignores).
            i_bl_driver_in__uA = (v_bl_clamp__V - v_bl_node.select(-1, 0)) * bl_driver_segment_g__uS
            i_sl_driver_in__uA = (v_sl_drive__V - v_sl_node.select(-2, 0)) * sl_driver_segment_g__uS
            v_bl_clamp__V, r_bl_driver_in__MOhm = self.bl_driver.solve_clamp(
                i_bl_driver_in__uA,
                bl_driver_snapshot,
                v_clamp_init__V=v_bl_clamp__V,
            )
            v_sl_drive__V, r_sl_driver_in__MOhm = self.sl_driver.solve_clamp(
                i_sl_driver_in__uA,
                sl_driver_snapshot,
                v_clamp_init__V=v_sl_drive__V,
            )
            v_bl_clamp_grid__V = v_bl_clamp__V.unsqueeze(-1)
            v_sl_drive_grid__V = v_sl_drive__V.unsqueeze(-2)

            f_bl_kcl = col_wire_kcl_residual(v_bl_node, v_bl_clamp_grid__V, bl_segment_g, i_cell)
            f_sl_kcl = row_wire_kcl_residual(v_sl_node, v_sl_drive_grid__V, sl_segment_g, -i_cell)
            # Both axial boundaries share the same Sherman-Morrison
            # rank-1 fold-in structure — only the wire axis (``dim``)
            # and the wire's driver-segment conductance differ.  For
            # an ideal driver ``r_driver_in__MOhm`` is the zero
            # tensor, so the rank-1 update vanishes and the step
            # degenerates to a plain tridiagonal solve — no
            # conditional in the kernel.  A future non-ideal driver
            # on either side lifts the same boundary into a proper
            # coupled Newton step on its (V_wire, V_drive) pair.
            dv_bl_node = self._solve_driver_newton_rank1(
                v_bl_node,
                f_bl_kcl,
                g_cell_eff,
                r_bl_driver_in__MOhm,
                driver_segment_g__uS=bl_driver_segment_g__uS,
                wire_diag=bl_wire_diag,
                wire_offdiag=bl_wire_offdiag,
                dim=-1,
            )
            dv_sl_node = self._solve_driver_newton_rank1(
                v_sl_node,
                f_sl_kcl,
                g_cell_eff,
                r_sl_driver_in__MOhm,
                driver_segment_g__uS=sl_driver_segment_g__uS,
                wire_diag=sl_wire_diag,
                wire_offdiag=sl_wire_offdiag,
                dim=-2,
            )
            v_bl_node = v_bl_node + dv_bl_node
            v_sl_node = v_sl_node + dv_sl_node

        # --- Driver currents from converged state ---
        # Each helper collapses the wire axis it integrates along.
        # Shape: i_bl_driver -> [..., num_col]
        i_bl_driver = col_driver_current(v_bl_node, v_bl_clamp_grid__V, bl_segment_g)
        # Shape: i_sl_driver -> [..., num_row]
        i_sl_driver = row_driver_current(v_sl_node, v_sl_drive_grid__V, sl_segment_g)
        return SolverResult(
            i_bl_driver=i_bl_driver,
            i_sl_driver=i_sl_driver,
            v_bl_node=v_bl_node,
            v_sl_node=v_sl_node,
            v_x_node=v_x_node,
            i_cell=i_cell,
            v_bl_clamp=v_bl_clamp__V,
            v_sl_drive=v_sl_drive__V,
        )

    # ---------------------------------------------------------------
    # Per-cell V_X / cell-current helpers
    # ---------------------------------------------------------------

    def _solve_cell_pade_warm_start(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid: Tensor,
        rram_state_g: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cold-start cell solve: linear divider seed + 1 Newton step + final eval.

        Used once per :meth:`solve` call to bootstrap the outer loop.
        The per-cell KCL is
        ``nmos.ids(wl, v_x_node, v_sl_node)
          = rram.i(v_bl_node − v_x_node, rram_snapshot)``.
        Letting ``v_cell_bl_to_sl = v_bl_node − v_sl_node`` and
        ``dc_nmos_seed.did_dvd__uS`` be the NMOS drain partial at
        the warm-start operating point ``v_d = v_bl_node``, the
        first-order Ohmic divider gives

            v_rram_drop_init = dc_nmos_seed.did_dvd · v_cell_bl_to_sl
                               / (dc_nmos_seed.did_dvd + g_static)

        so ``v_x_node_init = v_bl_node − v_rram_drop_init``.  The
        divider uses ``rram_state_g_static`` — the static
        programmed conductance — rather than the noisy snapshot:
        the warm start only needs a stable first-order operating
        point, and decoupling the seed from runtime noise keeps
        initialisation deterministic given the fabricated array
        (see ``temp/1t1r_solver.md``).  The Newton step that follows
        does use the noisy snapshot, so runtime noise still flows
        into the converged answer.

        Returns ``(i_cell, g_cell_eff, v_x_node)``.
        """
        # First-order (Ohmic) initial guess using the NMOS drain
        # partial at the warm-start operating point ``v_d = v_bl_node``.
        v_cell_bl_to_sl = v_bl_node - v_sl_node
        dc_nmos_seed = self.nmos.solve_dc(v_wl_drive_grid, v_bl_node, v_sl_node, nmos_snapshot)
        v_rram_drop_init = dc_nmos_seed.did_dvd__uS * v_cell_bl_to_sl / (dc_nmos_seed.did_dvd__uS + rram_state_g)
        v_x_node_init = v_bl_node - v_rram_drop_init
        return self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid,
            v_x_node_init,
            rram_snapshot,
            nmos_snapshot,
        )

    def _solve_cell_newton_warm_start(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid__V: Tensor,
        v_x_node_init: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cell solve seeded from a known-near ``v_x_node`` (inner outer-loop path).

        Skips the Padé divider — ``v_x_node_init`` is the previous
        outer iteration's converged ``v_x_node``, which is much
        closer to the new solution than the divider seed when
        ``v_bl_node`` / ``v_sl_node`` change incrementally between
        outer iters.  Otherwise identical to
        :meth:`_solve_cell_pade_warm_start`: one Newton step on the
        full cell KCL, then a final eval at the converged
        ``v_x_node`` for the returned ``i_cell`` and ``g_cell_eff``.
        """
        return self._solve_cell_newton_once(
            v_bl_node,
            v_sl_node,
            v_wl_drive_grid__V,
            v_x_node_init,
            rram_snapshot,
            nmos_snapshot,
        )

    def _solve_cell_newton_once(
        self,
        v_bl_node: Tensor,
        v_sl_node: Tensor,
        v_wl_drive_grid__V: Tensor,
        v_x_node_init: Tensor,
        rram_snapshot: RRAMSnapshot,
        nmos_snapshot: NMOSSnapshot,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One Newton step on the cell KCL from ``v_x_node_init`` + final eval.

        Shared body used by both the cold-start and warm-restart
        cell solves.  The cell KCL is
        ``f_cell_residual(v_x_node) = nmos.ids(wl, v_x_node, v_sl)
          − rram.i(v_bl_node − v_x_node)``, whose derivative is the
        sum of node partials (the chain factor ``−1`` on the RRAM
        term is absorbed by its sign-negated contribution to
        ``f_cell_residual``, so the derivative is a clean sum):

            dFcell_dVx = ∂I_nmos/∂v_d + ∂I_rram/∂v_rram

        After one Newton step we evaluate both devices at the
        converged ``v_x_node`` to compute the per-cell current and
        the Schur-complement effective conductance the outer wire
        Newton needs:

            g_cell_eff = (∂I_nmos/∂v_d · ∂I_rram/∂v)
                       / (∂I_nmos/∂v_d + ∂I_rram/∂v)
        """
        dc_nmos_at_init = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node_init, v_sl_node, nmos_snapshot)
        dc_rram_at_init = self.rram.solve_dc(v_bl_node - v_x_node_init, rram_snapshot)
        f_cell_residual = dc_nmos_at_init.ids__uA - dc_rram_at_init.i__uA
        dFcell_dVx = dc_nmos_at_init.did_dvd__uS + dc_rram_at_init.di_dv__uS
        v_x_node = v_x_node_init - f_cell_residual / dFcell_dVx

        # Final cell eval at the converged v_x_node.
        dc_nmos_final = self.nmos.solve_dc(v_wl_drive_grid__V, v_x_node, v_sl_node, nmos_snapshot)
        dc_rram_final = self.rram.solve_dc(v_bl_node - v_x_node, rram_snapshot)
        i_cell = dc_nmos_final.ids__uA
        g_cell_eff = (
            dc_nmos_final.did_dvd__uS * dc_rram_final.di_dv__uS / (dc_nmos_final.did_dvd__uS + dc_rram_final.di_dv__uS)
        )
        return i_cell, g_cell_eff, v_x_node

    # ---------------------------------------------------------------
    # Wire Newton helper
    # ---------------------------------------------------------------

    def _solve_driver_newton_rank1(
        self,
        v_node: Tensor,
        f_kcl: Tensor,
        g_cell_eff: Tensor,
        r_driver_in__MOhm: Tensor,
        *,
        driver_segment_g__uS: Tensor,
        wire_diag: Tensor,
        wire_offdiag: Tensor,
        dim: int,
    ) -> Tensor:
        """Solve a wire Newton system with a rank-1 clamp-driver fold-in.

        Generic over the active wire axis: ``dim=-1`` solves the BL
        wire (row axis), ``dim=-2`` solves the SL wire (col axis).
        Both axial boundaries share the same Sherman-Morrison
        rank-1 structure — the cell-side coupling enters with the
        same ``+g_cell_eff`` sign on both axes (BL: cell drains
        directly with ``∂i_cell/∂v_bl_node = +g_cell_eff``; SL:
        ``i_sl_port = −Σi_cell`` flips sign on the port-current
        identity and ``∂i_cell/∂v_sl_node = −g_cell_eff`` flips
        again on the cell partial, so the product
        ``∂i_port/∂v_sl_node = +g_cell_eff`` matches BL).  The wire
        instance and the active dim are the only per-call data the
        helper needs.

        Folds the driver's small-signal input impedance
        ``r_driver_in__MOhm`` (= ``∂v_drive / ∂i_port``) directly
        into the wire Jacobian, lifting the previous
        block-Gauss-Seidel "update the boundary, then run wire
        Newton" structure to a proper coupled Newton on
        ``(V_wire, V_drive)``.

        Math
        ----
        At node ``p`` (along ``dim``) the wire KCL residual is
        ``f_kcl_p = ±i_cell_p − 1{p=0}·(v_drive − v_node_0)·g_s[0]
                  + (wire-diff terms)``,
        so the only direct dependence of ``f_kcl`` on
        ``v_drive`` is in the ``p=0`` row with
        ``∂f_kcl_0/∂v_drive = −g_s[0]`` (the driver-segment
        conductance).  Linearising the clamp driver gives

            Δv_drive = r_driver_in__MOhm · Σ_q g_cell_eff_q · Δv_node_q,

        which lifts the wire Jacobian to

            g_jac_aug = g_jac_orig + u v^T
            u_p       = (−g_s[0] · r_driver_in__MOhm) · δ_{p,0}
            v_q       = g_cell_eff_q

        Sherman-Morrison turns this into one extra tridiagonal solve
        on top of the original ``−f_kcl → dv_base`` solve:

            T y = u,
            dv_node = dv_base
                      − v_rank1_response
                        · (v_q · dv_base / (1 + v_q · y))

        Note ``r_driver_in__MOhm < 0`` for an unclipped finite-gain
        driver (OpAmpTIA, CSA), so ``u_0 > 0`` — the rank-1 correction
        stiffens the ``p=0`` row, capturing the boundary's dynamic
        resistance.  For a rail-clipped OpAmpTIA, an ideal
        voltage-source clamp, or any boundary at zero local
        impedance, ``r_driver_in__MOhm = 0``, so ``u = 0`` and the
        Sherman-Morrison term vanishes — the wire Newton degenerates
        to the plain tridiagonal solve.

        The tridiagonal Jacobian templates ``wire_diag`` /
        ``wire_offdiag`` and the driver-segment conductance scalar
        ``driver_segment_g__uS`` are passed in already-derived (and
        invariant across the outer Newton iterations) by
        :meth:`solve`; this helper holds no wire-instance
        references so it stays trivially axis-agnostic.

        Args:
            v_node: Wire node voltages whose Newton increment we
                want — BL nodes for ``dim=-1``, SL nodes for
                ``dim=-2``.
            f_kcl: KCL residual for the corresponding wire.
            g_cell_eff: Per-cell Schur-complement effective
                conductance (positive; sign convention noted
                above).  Same tensor for both axes.
            r_driver_in__MOhm: Driver input impedance at this
                outer-iteration's port current (shape: along the
                opposite wire axis).
            driver_segment_g__uS: Driver-edge conductance scalar
                ``wire.segment_g__uS[0]`` for the active wire.
            wire_diag: Length-N diagonal template of the wire's
                tridiagonal Jacobian (excluding ``g_cell_eff``);
                derived once per :meth:`solve` call.
            wire_offdiag: Length-(N-1) off-diagonal template (same).
            dim: Wire axis on which to solve — must be negative
                (``-1`` for BL, ``-2`` for SL).

        Returns:
            Newton increment ``dv_node`` along ``dim``.
        """
        num_axis = v_node.shape[dim]
        shape_broadcast = [1] * v_node.ndim
        shape_broadcast[dim] = num_axis

        # ``sub`` / ``sup`` are length-N broadcasts of the pre-padded
        # wire off-diagonal, matching ``solve_tridiagonal``'s
        # same-shape convention.  sub[0] and sup[N-1] sit on positions
        # the algorithm never reads.
        g_jac_diag = wire_diag.view(shape_broadcast) + g_cell_eff
        sub_pattern = F.pad(wire_offdiag, (1, 0)).view(shape_broadcast)
        sup_pattern = F.pad(wire_offdiag, (0, 1)).view(shape_broadcast)
        sub = sub_pattern.expand_as(v_node)
        sup = sup_pattern.expand_as(v_node)

        # Original tridiagonal solve (Newton step without boundary coupling).
        dv_base = solve_tridiagonal(sub, g_jac_diag, sup, -f_kcl, dim=dim)

        # Rank-1 fold-in via Sherman-Morrison.  ``v_rank1_rhs`` is
        # non-zero only at index 0 along ``dim``; its scalar value
        # there is the left-vector coefficient
        # ``(−driver_segment_g__uS) · r_driver_in__MOhm``
        # (dimensionless).  Build the RHS by unsqueezing on ``dim``
        # to lift the scalar to a size-1 axis, then zero-padding
        # along ``dim`` out to ``num_axis``.  ``pad_spec`` follows
        # ``F.pad``'s last-dim-first convention: for ``dim=-d``
        # (d ≥ 1) we prepend ``d-1`` no-op pad pairs for the
        # trailing dims and end with ``(0, num_axis − 1)`` on
        # ``dim``.
        g_rank1_left_coeff = (-driver_segment_g__uS) * r_driver_in__MOhm
        pad_spec = (0, 0) * (-dim - 1) + (0, num_axis - 1)
        v_rank1_rhs = F.pad(g_rank1_left_coeff.unsqueeze(dim), pad_spec)

        v_rank1_response = solve_tridiagonal(sub, g_jac_diag, sup, v_rank1_rhs, dim=dim)
        sm_numerator = (g_cell_eff * dv_base).sum(dim=dim, keepdim=True)
        sm_denominator = 1.0 + (g_cell_eff * v_rank1_response).sum(dim=dim, keepdim=True)
        return dv_base - v_rank1_response * (sm_numerator / sm_denominator)
