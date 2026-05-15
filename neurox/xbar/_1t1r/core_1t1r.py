"""Shape-independent physical core for a 1T1R crossbar tile.

Architecture (see ``temp/1t1r_xbar.md``)::

    mapping / design layer  (Offset1T1RXbar, future Differential1T1RXbar)
        -> core_1t1r        (this module)
        -> readout          (neurox.xbar.readout)
        -> analog_mux       (neurox.analog.mux)
        -> adc              (neurox.analog.adc)
        -> digital post-process

``Core1T1R`` owns every shape-independent physical sub-component of one
1T1R tile: the RRAM / NMOS device modules, the two clamp drivers (BL
today: :class:`~neurox.analog.opamp_tia.OpAmpTIA`; SL today: the ideal
:class:`~neurox.analog.driver.Driver`), the three wire models
(BL / SL / WL), the WL DAC + WL decoder, plus the DC solver
(``NewtonRaphsonSolver1T1R``).  Its config carries the *static*
physical knobs (WL pulse width, four wire spacings); array shape is
**not** part of the config — it is read from ``w_phys.shape`` at
:meth:`fabricate` time, after which each device / clamp-driver module
fabricates its own internal buffers and the solver is constructed.

This separation lets one core be paired with different mapping layers
(offset coding today, differential coding later) without copying any
of the array physics; the mapping layer builds ``w_phys`` with the
target physical shape and calls ``core.fabricate(w_phys)`` once.

Per-core isolation
------------------
``Core1T1R`` takes **factories** for every device / circuit module it
owns (``rram_factory``, ``nmos_factory``, ``tia_factory``, plus the
peripheral analog factories) and instantiates them inside ``__init__``.
The clamp-driver slot is concrete (``OpAmpTIA`` today) rather than
Protocol-typed because :meth:`forward` also reads the OpAmpTIA-specific
``v_out__V`` from the post-solver re-call — see :meth:`forward`'s
``solver.solve_clamp`` vs. ``tia.solve_dc`` split.  No module instance is shared
across cores.  This guarantees that any per-fabrication or runtime
state stored inside a module belongs to exactly one core, eliminating
accidental cross-talk as ``RRAM`` / ``NMOS`` / ``OpAmpTIA`` evolve from
stateless tools into per-instance state holders (see
``temp/state_holding.md``).

State ownership (see ``temp/state_holding.md``)
-----------------------------------------------
Fabricated static state lives inside each owning module —
:class:`~neurox.device.RRAM` keeps ``state_g__uS`` after
:meth:`RRAM.fabricate`, :class:`~neurox.device.NMOS` keeps
``beta__uA_per_V2`` / ``vth__V``, the OpAmpTIA keeps ``opamp_gain``
and delegates to its internal NMOS submodule.  ``Core1T1R``
itself registers **no** device-level buffers; it just composes the
modules and pipes runtime snapshots through the solver.

Per-VMM runtime snapshots (``rram_snapshot``, ``nmos_snapshot``,
``bl_driver_snapshot``, ``sl_driver_snapshot``) are sampled once per
:meth:`forward` call and held as Python locals — they are **not**
registered as buffers.

Dynamic energy accounting
-------------------------
The core's :meth:`forward` returns a ``Core1T1ROutput`` carrying the
per-physical-column OpAmpTIA output voltage **and** the per-VMM dynamic
energy contributed by the array boundary itself:

* steady-state Joule dissipation over the WL pulse — covers both
  the BL and SL boundaries via ``v_clamp · i_driver`` on each axis.
* WL CV² (wire + per-cell ``C_gs``),
* BL node ground caps (wire + ``c_top``),
* node-X ground caps (``c_db`` + ``c_bot``),
* Miller-coupled ``C_gd`` between WL and node-X.

The SL clamp driver no longer reports a separate ``energy_per_op``
term — its Joule contribution is already captured by the supply-
power term ``v_sl_drive · i_sl_driver`` (per
``temp/sl_driver_solver.md``).

The WL DAC's per-element energy (``self.wl_dac.convert(x)``'s second
return) is deliberately **dropped**: it is already accounted for by the
``c_wl_per_row__fF · V_DD,WL²`` term inside the array-energy formula,
which captures the same physical charge.  Counting both would double-
book the WL switching energy.

ReadOut, AnalogMux, and ADC carry their own energy accounting through
the mapping layer's forward path; the core never sees them.

Definition domain
-----------------
After :meth:`fabricate`, both ``fabricated_col_num`` and
``fabricated_row_num`` must be ``> 1`` — the DC solver's wire model
requires at least one inter-node segment plus one driver segment on
each axis.  All four wire spacings are required strictly positive
(``> 0``) at config construction; the corresponding wire resistances
are checked again before the solver is built.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from neurox.analog import DAC, Decoder, Driver, OpAmpTIA
from neurox.device import NMOS, RRAM, Wire

from .solver_1t1r import NewtonRaphsonSolver1T1R, SolverResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Core1T1RConfig:
    """Shape-independent physical knobs for a 1T1R core.

    The wire-resistance fields are direct per-segment equivalent
    resistances (see ``temp/wire.md``): the orchestration layer
    composes these into the per-segment tensor handed to
    :meth:`Wire.fabricate`.  Geometric spacings are kept because the
    current energy path still folds in wire capacitance via a lumped
    ``c__fF_per_um × geometric_length`` approximation; resistance no
    longer depends on the spacings.

    Attributes:
        wl_pulse_length__ns: WL-high phase width ``t_pulse`` [ns].  Used
            as the integration window for the steady-state (resistive)
            array power.  ``0.0`` suppresses the thermal term; must be
            non-negative.
        sl_topology: Architectural axis of the source line.
            ``"row_shared"`` — SL is shared along the row axis,
            matching the standard 1T1R read/VMM topology that the
            current solver is built around.  ``"col_shared"`` is
            reserved for future column-shared SL layouts and is not
            implemented yet (:meth:`Core1T1R.fabricate` raises
            :class:`NotImplementedError` on encounter).  Surfacing
            this as an explicit field makes the SL assumption an
            architecture parameter instead of an implicit
            implementation detail (see ``temp/wire.md``).
        row_cell_space__um: Pitch between adjacent cells along the
            row axis [um].  Only drives the wire-capacitance energy
            estimate.
        row_first_space__um: Distance from the row-aligned driver to
            the first cell [um].  Only drives the wire-capacitance
            energy estimate.
        col_cell_space__um: Pitch between adjacent cells along the
            column axis [um].  Only drives the wire-capacitance
            energy estimate.
        col_first_space__um: Distance from the BL driver to the first
            cell [um].  Only drives the wire-capacitance energy
            estimate.
        bl_r_driver_to_first__MOhm: BL driver-to-first-cell segment
            equivalent resistance [MOhm].  Direct equivalent value
            — users may compose it from a uniform resistivity
            (``r_per_um × col_first_space``) or from a layout-extracted
            total that already accounts for vias, metal-stack
            transitions, and local widening; the wire itself sees
            only this scalar (see ``temp/wire.md``).
        bl_r_cell_to_cell__MOhm: BL inter-cell segment equivalent
            resistance [MOhm].
        sl_r_driver_to_first__MOhm: SL driver-to-first-cell segment
            equivalent resistance [MOhm].
        sl_r_cell_to_cell__MOhm: SL inter-cell segment equivalent
            resistance [MOhm].
    """

    wl_pulse_length__ns: float
    sl_topology: Literal["row_shared", "col_shared"]

    row_cell_space__um: float
    row_first_space__um: float
    col_cell_space__um: float
    col_first_space__um: float

    bl_r_driver_to_first__MOhm: float
    bl_r_cell_to_cell__MOhm: float
    sl_r_driver_to_first__MOhm: float
    sl_r_cell_to_cell__MOhm: float

    def __post_init__(self) -> None:
        if not (self.wl_pulse_length__ns >= 0.0):
            raise ValueError(f"require: wl_pulse_length__ns ({self.wl_pulse_length__ns}) >= 0.0")
        if self.sl_topology not in ("row_shared", "col_shared"):
            raise ValueError(f"require: sl_topology ({self.sl_topology!r}) in ('row_shared', 'col_shared')")
        if not (self.row_cell_space__um > 0.0):
            raise ValueError(f"require: row_cell_space__um ({self.row_cell_space__um}) > 0.0")
        if not (self.row_first_space__um > 0.0):
            raise ValueError(f"require: row_first_space__um ({self.row_first_space__um}) > 0.0")
        if not (self.col_cell_space__um > 0.0):
            raise ValueError(f"require: col_cell_space__um ({self.col_cell_space__um}) > 0.0")
        if not (self.col_first_space__um > 0.0):
            raise ValueError(f"require: col_first_space__um ({self.col_first_space__um}) > 0.0")
        for name in (
            "bl_r_driver_to_first__MOhm",
            "bl_r_cell_to_cell__MOhm",
            "sl_r_driver_to_first__MOhm",
            "sl_r_cell_to_cell__MOhm",
        ):
            value = getattr(self, name)
            if not (value > 0.0):
                raise ValueError(f"require: {name} ({value}) > 0.0")


# ---------------------------------------------------------------------------
# Forward output container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Core1T1ROutput:
    """Per-VMM analog output of a fabricated 1T1R core.

    Dynamic energy is no longer carried on this dataclass — the
    physical leaves inside the core (RRAM, NMOS, OpAmpTIA, DAC, …)
    emit their own profiler events through the side-channel logger.

    Attributes:
        v_out_phys: OpAmpTIA output voltage per physical column [V].  Shape:
            ``[*batch, phys_col_num]``.  Downstream readout consumes
            this directly without ever seeing the underlying solver
            state.
    """

    v_out_phys: Tensor


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


class Core1T1R(nn.Module):
    """Shape-independent 1T1R core: array + clamp driver + DC solver.

    Holds every per-tile physical sub-component that is independent of
    the eventual array shape.  All device / circuit modules are
    instantiated from factories at ``__init__`` time — no instance is
    shared with another core.

    The shape is supplied later by the mapping layer through
    :meth:`fabricate`'s ``w_phys`` argument — the core reads
    ``phys_col_num`` and ``row_num`` straight off ``w_phys.shape``,
    delegates fabricate to each owned module (which registers its own
    buffers internally), recomputes the four capacitance scalars used
    by the dynamic-energy formula, and binds the solver.

    Lifecycle:

    1. ``__init__(cfg, *_factory)`` — builds each module by calling its
       factory, derives wire resistances, caches ``v_dd_wl__V`` from
       the WL DAC.  No solver yet, no per-cell buffers yet.
    2. ``fabricate(w_phys)`` — reads shape, calls each module's
       ``fabricate(...)``, recomputes the four capacitance scalars,
       builds the solver.
    3. ``forward(x)`` — samples per-VMM runtime snapshots, runs the
       DAC + solver-owned BL/SL boundary solve + Newton-Raphson +
       array-energy aggregation, and re-invokes the BL clamp driver
       with the same ``bl_driver_snapshot`` to obtain ``v_out_phys``.
       Returns ``Core1T1ROutput``.

    Owned submodules (built from factories in :meth:`__init__`):
        ``rram``, ``nmos``, ``tia``, ``sl_driver``, ``wl_decoder``,
        ``wl_dac``.  The three wire models are referenced via plain
        attributes — :class:`Wire` has no buffers worth moving across
        devices, so submodule registration would only add ``.to()``
        bookkeeping for nothing.

    Owned submodule (set in :meth:`fabricate`):
        ``solver`` — the DC :class:`NewtonRaphsonSolver1T1R`.

    Cached scalars (constant across fabrications, set in ``__init__``):
        ``v_dd_wl__V``.

    Cached scalars (depend on fabricated shape, set in :meth:`fabricate`):
        ``c_wl_per_row__fF``, ``c_bl_per_node__fF``,
        ``c_x_per_cell__fF``, ``c_gd_per_cell__fF``.

    Wire state (depends on fabricated shape, owned by the wire
    modules themselves after :meth:`fabricate` per
    ``temp/wire.md``):
        ``bl_wire.{segment_r__MOhm, segment_g__uS}`` and the SL
        equivalents.  The per-segment resistance tensor is built
        directly by :meth:`Core1T1R.fabricate` from the
        :class:`Core1T1RConfig` ``*_r_*__MOhm`` fields — no length ×
        resistivity step lives inside :class:`Wire`.

    Capability accessors:
        ``w_states`` (``= rram.num_states``), ``x_states`` (= 2 for the
        binary WL gating used by every 1T1R tile today).

    Args:
        cfg: Shape-independent physical knobs.
        rram_factory: Zero-argument factory building this core's RRAM
            instance.  Each :class:`Core1T1R` owns a fresh RRAM.
        nmos_factory: Zero-argument factory building the array-access
            NMOS instance.
        tia_factory: Zero-argument factory building this core's BL
            clamp driver — a :class:`~neurox.analog.opamp_tia.OpAmpTIA` today.
            The factory is responsible for wiring in any nested
            sub-modules (OpAmpTIA's internal pseudo-resistor NMOS).
            Typed concretely (rather than as
            ``Callable[[], ClampDriver]``) because :meth:`forward`
            reads the OpAmpTIA-specific ``v_out__V`` off the post-solver
            re-call; the narrower
            :class:`~neurox.analog.ClampDriver` protocol is only
            used inside the solver itself.
        sl_wire_factory / bl_wire_factory / wl_wire_factory:
            Zero-argument factories building this core's BL / SL / WL
            :class:`~neurox.device.Wire` instances.  Each
            :class:`Core1T1R` owns a fresh, fabricated-per-core wire
            triple — per-segment resistance tensors are built locally
            from :class:`Core1T1RConfig`'s direct-resistance fields
            (see ``temp/wire.md``).
        sl_driver_factory: Factory building the SL clamp driver.  The
            SL driver satisfies the :class:`~neurox.analog.ClampDriver`
            protocol — the solver consumes it through
            :meth:`~neurox.analog.ClampDriver.solve_clamp` on every
            outer Newton iteration, exactly the same way the BL slot
            is consumed.  Today this is the ideal
            :class:`~neurox.analog.driver.Driver`; a future non-ideal
            source-side driver drops in here without changing the
            solver's public structure.
        wl_decoder_factory: Factory building the WL decoder.  Built for
            PPA accounting only; not invoked in :meth:`forward` — a
            future commit will wire it in.
        wl_dac_factory: Factory building the WL DAC.
        dtype: Floating-point dtype propagated to the owned device /
            wire modules through their factories; the core itself
            registers no per-fabrication buffers (wire state lives on
            the wire instances).
    """

    def __init__(
        self,
        cfg: Core1T1RConfig,
        *,
        name: str = "",
        rram_factory: Callable[..., RRAM],
        nmos_factory: Callable[..., NMOS],
        tia_factory: Callable[..., OpAmpTIA],
        sl_wire_factory: Callable[..., Wire],
        bl_wire_factory: Callable[..., Wire],
        wl_wire_factory: Callable[..., Wire],
        sl_driver_factory: Callable[..., Driver],
        wl_decoder_factory: Callable[..., Decoder],
        wl_dac_factory: Callable[..., DAC],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self._neurox_name = name
        self.cfg = cfg
        self.dtype = dtype

        # Per-instance device / clamp-driver / wire modules — each
        # core owns its own.  No singleton sharing across cores; the
        # wire state is per-fabrication so every core needs a fresh
        # wire triple (see ``temp/state_holding.md`` and
        # ``temp/wire.md``).  Child names cascade from this core's
        # hierarchical name so the profiler emits ``<core>.<child>``
        # events (e.g. ``fc1.macro.xbar.core.tia``).
        prefix = name + "." if name else ""
        self.rram = rram_factory(name=f"{prefix}rram")
        self.nmos = nmos_factory(name=f"{prefix}nmos")
        self.tia = tia_factory(name=f"{prefix}tia")
        self.sl_wire = sl_wire_factory(name=f"{prefix}sl_wire")
        self.bl_wire = bl_wire_factory(name=f"{prefix}bl_wire")
        self.wl_wire = wl_wire_factory(name=f"{prefix}wl_wire")

        # Build the peripheral analog blocks once per core.
        self.sl_driver = sl_driver_factory(name=f"{prefix}sl_driver")
        self.wl_decoder = wl_decoder_factory(name=f"{prefix}wl_decoder")
        self.wl_dac = wl_dac_factory(name=f"{prefix}wl_dac")

        # Per-segment wire resistance tensors are built in
        # :meth:`fabricate` directly from :class:`Core1T1RConfig`'s
        # direct-equivalent resistance fields and handed to
        # ``Wire.fabricate(segment_r__MOhm)`` (see ``temp/wire.md``).
        # ``WireConfig`` only carries the lumped ``c__fF_per_um`` used
        # by the energy path; the wire never sees lengths.

        # V_DD,WL — single source of truth is the WL DAC's on-level
        # (``code_to_signal[1]`` for the 2-state DAC the 1T1R tile uses).
        self.v_dd_wl__V: float = float(self.wl_dac.code_to_signal[1].item())

        # Capability surface; constant for the lifetime of the core.
        self.w_states: int = self.rram.num_states
        self.x_states: int = 2

        # Shape-dependent capacitances and the solver are set up in
        # ``fabricate`` once the physical shape is known.  Wire state
        # lives on the wire instances themselves (see
        # ``temp/wire.md``).
        self.solver: NewtonRaphsonSolver1T1R

        self.fabricated_row_num: int = 0
        self.fabricated_col_num: int = 0
        self.c_wl_per_row__fF: float = 0.0
        self.c_bl_per_node__fF: float = 0.0
        self.c_x_per_cell__fF: float = 0.0
        self.c_gd_per_cell__fF: float = 0.0

    # -----------------------------------------------------------------
    # Fabrication
    # -----------------------------------------------------------------

    def fabricate(self, w_phys: Tensor) -> None:
        """Program per-cell buffers from a physically-shaped weight tensor.

        ``w_phys`` is expected to be the **physical** layout produced by
        the mapping layer — i.e. ref columns already scattered for
        ``Offset1T1RXbar`` (offset state shift applied) or pos/neg
        column pairs already laid out for a future
        ``Differential1T1RXbar``.  This method does no mapping of its
        own; it reads the shape, delegates fabrication to each owned
        module (which registers its own buffers internally), and binds
        the solver.

        Args:
            w_phys: Physical state indices.  Shape:
                ``[..., phys_col_num, row_num]`` (the trailing two dims
                are the only ones the solver cares about; any leading
                batch is preserved on the RRAM's internal
                ``state_g__uS`` buffer).
        """
        *_, phys_col_num, row_num = w_phys.shape
        if not (phys_col_num > 1):
            raise ValueError(f"require: phys_col_num ({phys_col_num}) > 1")
        if not (row_num > 1):
            raise ValueError(f"require: row_num ({row_num}) > 1")
        # SL topology gate.  The 1D-ladder solver is built around the
        # ``row_shared`` source line; ``col_shared`` would invert the
        # wire-axis bookkeeping in :class:`NewtonRaphsonSolver1T1R` and
        # is reserved for a future build (see ``temp/wire.md``).
        if self.cfg.sl_topology != "row_shared":
            raise NotImplementedError(
                f"sl_topology={self.cfg.sl_topology!r} is reserved for a future "
                "build; only 'row_shared' is implemented today (see temp/wire.md)"
            )

        # Delegate fabrication to each owned module.  Each module owns
        # its own buffers; the core sees only the runtime-state
        # contract.  RRAM's stateful programming entry point is
        # ``program(state)`` (no separate ``fabricate``).
        self.rram.program(w_phys)
        self.nmos.fabricate(w_phys.shape)
        self.tia.fabricate((phys_col_num,))

        # Capacitance scalars — see ``doc/design/xbar_1t1r_energy.md``.
        # All four are precomputed Python floats so the forward kernel
        # sees them as compile-time constants.  Wire capacitance is
        # the only place spacings are still consumed; per-segment
        # capacitance is intentionally deferred (see ``temp/wire.md``).
        wl_wire_cap__fF = self.wl_wire.c__fF_per_um * (
            self.cfg.row_first_space__um + (phys_col_num - 1) * self.cfg.row_cell_space__um
        )
        self.c_wl_per_row__fF = wl_wire_cap__fF + phys_col_num * self.nmos.c_gs__fF
        bl_wire_total__fF = self.bl_wire.c__fF_per_um * (
            self.cfg.col_first_space__um + (row_num - 1) * self.cfg.col_cell_space__um
        )
        self.c_bl_per_node__fF = bl_wire_total__fF / row_num + self.rram.c_top__fF
        self.c_x_per_cell__fF = self.nmos.c_db__fF + self.rram.c_bot__fF
        self.c_gd_per_cell__fF = self.nmos.c_gd__fF

        # Per-line segment-resistance tensors handed straight to
        # :meth:`Wire.fabricate`.  Index 0 is the driver-to-first
        # segment; indices ``[1:]`` are the inter-cell segments.  The
        # tensor is constructed from direct equivalent-resistance
        # config fields (see ``temp/wire.md``); no length × r-per-um
        # derivation lives inside :class:`Wire` anymore.
        bl_segment_r__MOhm = torch.tensor(
            [self.cfg.bl_r_driver_to_first__MOhm] + [self.cfg.bl_r_cell_to_cell__MOhm] * (row_num - 1),
            dtype=self.dtype,
        )
        sl_segment_r__MOhm = torch.tensor(
            [self.cfg.sl_r_driver_to_first__MOhm] + [self.cfg.sl_r_cell_to_cell__MOhm] * (phys_col_num - 1),
            dtype=self.dtype,
        )
        self.bl_wire.fabricate(bl_segment_r__MOhm)
        self.sl_wire.fabricate(sl_segment_r__MOhm)

        # Bind the DC solver.  The solver's geometry comes from the
        # runtime input shape on each ``solve`` call; we only thread
        # the device modules + the two clamp drivers + fabricated
        # wires here.  Both clamp-driver slots are Protocol-typed
        # inside the solver (:class:`~neurox.analog.ClampDriver`), so
        # a future CSA driver — or any other non-ideal source-side
        # implementation — could plug in here without touching the
        # solver.  :class:`Core1T1R` keeps the concrete handle on
        # ``self.tia`` for the post-solver ``solve_dc`` re-call below
        # that recovers ``v_out__V`` for the readout path.
        self.solver = NewtonRaphsonSolver1T1R(
            rram=self.rram,
            nmos=self.nmos,
            bl_driver=self.tia,
            sl_driver=self.sl_driver,
            bl_wire=self.bl_wire,
            sl_wire=self.sl_wire,
        )

        self.fabricated_col_num = phys_col_num
        self.fabricated_row_num = row_num

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, x: Tensor) -> Core1T1ROutput:
        """Run one VMM on the physical 1T1R core.

        Args:
            x: Binary WL gate signal aligned to the input cell axis.
                Shape: broadcastable to ``[..., 1, row_num]``.  The
                size-1 slot is the per-physical-column broadcast
                axis the inner solver / OpAmpTIA expect; any
                caller-side macro axes ride along as opaque
                broadcast leading dims.

        Returns:
            :class:`Core1T1ROutput` carrying the per-physical-column
            OpAmpTIA output voltage ``v_out_phys``.  Dynamic energy
            is no longer in the return value — every leaf emits its
            own profiler event through the side channel.
        """
        # Derive the execution shape from the fabricated physical
        # layout + the input.  We use the RRAM's internal state shape
        # so the broadcast contract follows the device that drives
        # the per-element noise sampling.
        # Shape: full_shape = [*batch, phys_col_num, row_num].
        full_shape = torch.broadcast_shapes(self.rram.state_g__uS.shape, x.shape)
        *batch, _phys_col_num, row_num = full_shape

        # One-shot per-VMM runtime sampling (see ``temp/state_holding.md``).
        # The runtime objects are local Python values — never persisted —
        # and threaded unchanged through the solver and the post-solver
        # OpAmpTIA re-call.
        # Shape: rram_snapshot / nmos_snapshot fields -> [..., phys_col_num, row_num].
        rram_snapshot = self.rram.snapshot(shape=full_shape)
        nmos_snapshot = self.nmos.snapshot(shape=full_shape)
        # Shape: bl_driver_snapshot field -> [..., phys_col_num].
        bl_driver_snapshot = self.tia.snapshot(shape=(*batch, self.fabricated_col_num))
        # Shape: sl_driver_snapshot field -> [..., row_num].
        sl_driver_snapshot = self.sl_driver.snapshot(shape=(*batch, row_num))

        # Per-row WL drive voltage.  The DAC's ``convert`` returns
        # ``v__V`` directly; switching energy / latency flow through
        # the profiler side channel from inside the DAC.
        # Shape: x -> [..., 1, row_num] (broadcast expand against full_shape).
        x = x.expand(*batch, 1, row_num)
        # Shape: wl_logic -> [..., row_num].
        wl_logic = x.squeeze(-2)
        # Shape: v_wl_drive__V -> [..., 1, row_num].
        v_wl_drive__V = self.wl_dac.convert(x)

        # DC solve — pure tensor pipeline, no Python control flow on
        # tensor values, fully ``@torch.compile``-friendly.  Wire
        # state is read by the solver from ``self.bl_wire`` /
        # ``self.sl_wire`` (per ``temp/wire.md``).  The SL boundary
        # voltage is no longer precomputed here — the solver owns
        # both BL and SL clamp boundaries and returns the converged
        # ``v_sl_drive`` through :class:`SolverResult`.
        # Shape: solver_result fields -> see SolverResult docstring
        # (node tensors are ``[..., phys_col_num, row_num]``, boundary
        # currents / clamps are ``[..., phys_col_num]`` or
        # ``[..., row_num]``).
        solver_result = self.solver.solve(
            v_wl_drive__V,
            rram_snapshot=rram_snapshot,
            nmos_snapshot=nmos_snapshot,
            bl_driver_snapshot=bl_driver_snapshot,
            sl_driver_snapshot=sl_driver_snapshot,
        )

        # Core dynamic energy: array CV² / Joule emitted as a side-
        # channel event on ``self.tia`` (the OpAmpTIA is the natural
        # owner of the BL boundary's per-VMM array energy).  See
        # ``doc/design/xbar_1t1r_energy.md``.
        # Shape: array_energy__fJ -> [*batch].
        array_energy__fJ = self._compute_array_energy__fJ(
            solver_result=solver_result,
            wl_logic=wl_logic,
        )
        self.tia._log_dynamic(array_energy__fJ, self.tia.cfg.latency_per_op__ns)

        # ``SolverResult`` does not carry ``v_out`` — the readout-path
        # quantity is concrete-clamp-driver-specific.  Re-evaluate the
        # OpAmpTIA once at the solver's converged port current using the
        # *same* ``bl_driver_snapshot`` so the post-solver state is
        # consistent with the solved boundary.  ``OpAmpTIA.solve_dc`` is
        # the richer concrete-class entry point returning
        # :class:`OpAmpTIADC` — the Protocol-level :meth:`solve_clamp`
        # would only return the ``(v_clamp, dVclamp_dI)`` tuple, but
        # the readout path also needs ``v_out__V``.  The converged
        # ``solver_result.i_bl_driver`` is the same boundary-KCL
        # ``delta_v · g`` quantity the solver fed into every inner
        # ``solve_clamp`` call, so we feed it directly here — no
        # ``i_cell.sum(-1)`` shortcut (that identity is only true on
        # the 1-D BL ladder and would silently break under richer
        # topologies; see ``temp/solver.md``).
        clamp_dc = self.tia.solve_dc(
            solver_result.i_bl_driver,
            bl_driver_snapshot,
            v_clamp_init__V=solver_result.v_bl_clamp,
        )
        # Shape: v_out_phys -> [..., phys_col_num].
        v_out_phys = clamp_dc.v_out__V

        return Core1T1ROutput(v_out_phys=v_out_phys)

    # -----------------------------------------------------------------
    # Dynamic-energy aggregation
    # -----------------------------------------------------------------

    def _compute_array_energy__fJ(
        self,
        solver_result: SolverResult,
        wl_logic: Tensor,
    ) -> Tensor:
        """Compute per-VMM array-internal dynamic energy.

        Implements the Miller-reduced physical model documented in
        ``doc/design/xbar_1t1r_energy.md``.  Three of the NMOS's four
        terminal caps are pure ground caps and have been folded into
        the scalars precomputed in :meth:`fabricate`; ``C_gd`` is
        Miller-reduced on the fly using the solver-computed per-cell
        ``V_X^(1)`` and the per-row ``V_WL^(1) = V_DD,WL · wl_logic``.

        Per-column BL clamp voltage and per-row SL drive voltage come
        from the converged solver state: under heavy column draw the
        BL OpAmpTIA's finite op-amp gain pulls the clamp below ``v_ref``;
        any future non-ideal SL driver would equally shift its drive
        voltage with row current.  Every term sees the dynamic
        per-port value rather than a tile-wide scalar.

        Args:
            solver_result: Converged DC solve from
                :meth:`NewtonRaphsonSolver1T1R.solve`.  Carries both
                boundary clamp voltages (``v_bl_clamp`` /
                ``v_sl_drive``) used by the supply-power term.
            wl_logic: Binary WL-gate signal per row (1 = row pulses,
                0 = row idle).  Shape ``[*batch, row_num]``.

        Returns:
            Per-array dynamic energy [fJ].  Shape ``[*batch]``.
        """
        # --- solver outputs ---
        # Shape: i_bl_driver__uA -> [..., phys_col_num]
        i_bl_driver__uA = solver_result.i_bl_driver
        # Shape: i_sl_driver__uA -> [..., row_num]
        i_sl_driver__uA = solver_result.i_sl_driver
        # Shape: v_bl_node__V -> [..., phys_col_num, row_num]
        v_bl_node__V = solver_result.v_bl_node
        # Shape: v_x_node__V -> [..., phys_col_num, row_num]
        v_x_node__V = solver_result.v_x_node
        # Shape: v_bl_clamp__V -> [..., phys_col_num]
        v_bl_clamp__V = solver_result.v_bl_clamp
        # Shape: v_sl_drive__V -> [..., row_num]
        v_sl_drive__V = solver_result.v_sl_drive

        # --- broadcasting setup ---
        v_dd_wl = self.v_dd_wl__V
        # Shape: v_wl_state1__V -> [..., 1, row_num]
        v_wl_state1__V = (v_dd_wl * wl_logic).unsqueeze(-2)
        # Shape: v_clamp__V -> [..., phys_col_num, 1]
        v_clamp__V = v_bl_clamp__V.unsqueeze(-1)

        # Every per-term ``e_*__fJ`` below collapses to ``[*batch]``;
        # leading shape comments only flag the changed (reduced) axes.
        # --- E_thermal: steady-state Joule dissipation over the WL pulse ---
        # Supply-power balance:
        #   P = V_bl · I_bl_into_array + V_sl · I_sl_into_array
        # (driver-into-wire sign convention).  Unit: uW · ns = fJ.
        # Shape: array_power__uW -> [...]
        array_power__uW = torch.sum(v_bl_clamp__V * i_bl_driver__uA, dim=-1) + torch.sum(
            v_sl_drive__V * i_sl_driver__uA, dim=-1
        )
        e_thermal__fJ = array_power__uW * self.cfg.wl_pulse_length__ns

        # --- E_WL_drive: wire + C_gs ground caps at WL ---
        # Inactive rows contribute zero (V_WL^(1) = 0 for binary DAC).
        # Unit: fF · V² = fJ.
        # Shape: e_wl_ground__fJ -> [...]
        e_wl_ground__fJ = wl_logic.sum(dim=-1) * self.c_wl_per_row__fF * v_dd_wl * v_dd_wl

        # --- E_BL_recover at Node X: C_db + c_bot ground caps ---
        # E_per_cell = V_BL_clamp · C_X · (V_BL_clamp - V_X^(1)).
        delta_v_x__V = v_clamp__V - v_x_node__V
        # Shape: e_bl_x_ground__fJ -> [...]
        e_bl_x_ground__fJ = (v_clamp__V.squeeze(-1) * self.c_x_per_cell__fF * delta_v_x__V.sum(dim=-1)).sum(dim=-1)

        # --- E_BL_recover at BL nodes: wire + c_top ground caps ---
        # E_per_node = V_BL_clamp · C_BL_node · (V_BL_clamp - V_BL^(1)).
        delta_v_bl__V = v_clamp__V - v_bl_node__V
        # Shape: e_bl_node__fJ -> [...]
        e_bl_node__fJ = (v_clamp__V * self.c_bl_per_node__fF * delta_v_bl__V).sum(dim=(-2, -1))

        # --- E_Cgd: Miller-coupled gate-drain cap ---
        # Combined WL + BL supply energy per cell:
        #   E = C_gd · (V_WL^(1) + V_BL_clamp) · (V_WL^(1) + V_BL_clamp − V_X^(1))
        sum_voltage__V = v_wl_state1__V + v_clamp__V
        delta_v_miller__V = sum_voltage__V - v_x_node__V
        # Shape: e_cgd__fJ -> [...]
        e_cgd__fJ = (sum_voltage__V * self.c_gd_per_cell__fF * delta_v_miller__V).sum(dim=(-2, -1))

        return e_thermal__fJ + e_wl_ground__fJ + e_bl_x_ground__fJ + e_bl_node__fJ + e_cgd__fJ
