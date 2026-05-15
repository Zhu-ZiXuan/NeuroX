"""RRAM (Resistive Random Access Memory) conductive-filament device model.

Physical model overview
-----------------------
Conductance states map a discrete integer index to one of ``N`` ideal levels
``g_0 < g_1 < ... < g_{N-1}`` (in uS).  Four non-ideality layers are stacked
on top of that ideal map:

**Programming variation (Gamma noise)**
    After mapping a state index to its ideal level ``g``, a state-dependent
    Gamma multiplicative noise is applied.  The Gamma shape parameter ``k``
    is a linear function of the normalised conductance
    ``x_norm = (g - g_min) / (g_max - g_min)``:

        k(x_norm) = k_slope * x_norm + k_intercept

    Low-conductance (thin-filament) cells have small ``k`` → heavy tail.
    High-conductance cells have large ``k`` → tighter distribution.
    The scale parameter ``theta`` is shared across all states.
    The sample is normalised to unit mean so ``E[g_noisy] = g_ideal``.

**Temporal drift (power-law)**
    Conductance decays as:

        g(t) = g_prog * (t / t0)^(-nu)

    where ``nu = drift_decay_rate`` and ``t0 = drift_t0`` [s].
    Drift is applied only when ``t_elapsed > t0`` and ``nu > 0``.

**Read noise**
    Two additive read-time perturbations are applied in sequence:

    * *Random telegraph noise (RTN)*: each cell independently enters the
      high-RTN state with probability ``p_high_state``; when active, a
      signed perturbation with amplitude drawn from
      ``N(amplitude_mean, amplitude_std^2)`` is added.
    * *Thermal noise*: additive Gaussian ``N(0, sigma^2)`` [uS].

**I-V non-linearity (sinh model)**
    Device current [uA] for a read voltage ``v`` [V] at conductance ``g`` [uS]:

        I = g * v                               (alpha == 0, Ohmic)
        I = g * sinh(alpha * v) / alpha         (alpha > 0)

    ``alpha`` [1/V] controls the degree of super-Ohmic non-linearity.

**Stuck-at faults**
    Cells are independently frozen at ``g_min`` (stuck-at-LRS) or ``g_max``
    (stuck-at-HRS) with configurable probabilities.

All non-idealities are implemented as closed-form tensor operations compatible
with ``torch.compile``.  ``program`` is intentionally invoked outside
``torch.compile`` (it runs once at fabricate time) because the Gamma sampler
requires float32 and is not yet fully dynamo-traceable; ``snapshot`` is
called inside the xbar's compiled forward and returns a fresh local
snapshot that the surrounding kernel can fuse and reclaim.

Solver-facing API
-----------------
For Newton-style circuit solvers running under ``@torch.compile``,
:meth:`solve_dc` returns the working-point current **and** the local
differential conductance ``∂I/∂V`` in a single call so the
sinh / cosh intermediates can be fused once per cell.  The result is
a :class:`RRAMDC` frozen dataclass — chosen over ``NamedTuple``
because the latter currently hits known Dynamo tracing issues in
this codebase.

State ownership
---------------
:class:`RRAM` **internally owns** its fabricated post-program
conductance.  Lifecycle:

1. :meth:`program(state, t_elapsed=0.0)` — maps state indices to
   post-program conductance (Gamma + drift + stuck-at + clamp) and
   registers the result as ``self.state_g__uS`` (non-persistent
   buffer).  Stateful: each call replaces the buffer with a fresh
   sample.
2. :meth:`snapshot(shape=...)` — expands ``state_g__uS`` to
   the requested execution shape, applies read-time RTN + thermal
   noise, and returns a :class:`RRAMSnapshot` snapshot.
3. :meth:`solve_dc(v, runtime)` — evaluates the sinh I-V at any
   operating point using the runtime snapshot.

Higher layers (``Core1T1R``, solver) see only the runtime dataclass;
the post-program tensor stays inside the module.  Per-VMM runtime
snapshots are **not** registered as buffers — they are local objects
with a single-VMM lifetime per
``temp/state_holding.md``.
"""

from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import (
    StateDependentGammaConfig,
    StuckAtFaultConfig,
    TelegraphConfig,
    apply_gaussian,
    apply_state_dependent_gamma,
    apply_stuck_at_fault,
    apply_telegraph_noise,
)


@dataclass(frozen=True)
class RRAMConfig:
    """Immutable configuration for a single RRAM device.

    Each noise sub-config is optional: when it is ``None`` the
    corresponding noise layer is skipped entirely by the
    ``apply_*`` functions.  Whether a noise layer fires is therefore
    controlled exclusively by the config file — omit the sub-table to
    disable that layer.

    Temporal drift is a deterministic decay (not a stochastic noise) and
    is always configured inline; set ``drift_decay_rate = 0.0`` to
    suppress it.

    Attributes:
        state_to_g__uS: Ideal conductance lookup table [uS], one entry per
            discrete state, strictly increasing.  Index 0 is the lowest
            conductance (HRS); index ``N-1`` is the highest (LRS).
        nonlinearity_alpha: Sinh I-V coefficient ``alpha`` [1/V].
            Set to ``0.0`` for a purely Ohmic (linear) response.
        drift_decay_rate: Power-law exponent ``nu`` >= 0.  ``0.0`` disables
            drift.
        drift_t0: Reference time ``t0`` [s] anchoring the drift power
            law.  Must be non-negative; drift is only applied when
            ``t_elapsed > drift_t0``.
        prog_gamma: State-dependent Gamma parameters for programming
            variation.  ``None`` skips programming noise.
        read_telegraph: RTN parameters applied at read.  ``None`` skips RTN.
        read_thermal: Additive Gaussian thermal noise [uS] applied at read.
            ``None`` skips thermal noise.
        stuck_at: Stuck-at fault parameters.  ``None`` skips stuck-at faults.
        c_top__fF: Lumped BL-side (top electrode) parasitic capacitance per
            cell [fF].  Contributes to the per-BL-node load
            ``C_BL_node = C_wire_bl_node + c_top__fF`` that the BL clamp
            source must refill on the WL-off edge.
        c_bot__fF: Lumped Node-X-side (bottom electrode) parasitic
            capacitance per cell [fF].  Contributes to the per-cell
            ``C_X = switch.c_db__fF + c_bot__fF`` pulled down during the
            WL-high window and refilled by BL clamp on WL-off.
    """

    state_to_g_map__uS: list[float]

    nonlinearity_alpha: float

    drift_decay_rate: float
    drift_t0: float

    c_top__fF: float
    c_bot__fF: float

    prog_gamma: StateDependentGammaConfig | None = None

    read_telegraph: TelegraphConfig | None = None
    read_thermal: float | None = None

    stuck_at: StuckAtFaultConfig | None = None

    @property
    def num_states(self) -> int:
        """Number of available discrete conductance states."""
        return len(self.state_to_g_map__uS)

    @property
    def g_min__uS(self) -> float:
        """Minimum ideal conductance derived from state levels."""
        return self.state_to_g_map__uS[0]

    @property
    def g_max__uS(self) -> float:
        """Maximum ideal conductance derived from state levels."""
        return self.state_to_g_map__uS[-1]

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_states < 2:
            raise ValueError(f"len(state_to_g__uS) ({self.num_states}) < 2")
        if any(curr <= prev for prev, curr in pairwise(self.state_to_g_map__uS)):
            raise ValueError(f"state_to_g__uS ({self.state_to_g_map__uS}) must be strictly increasing")
        if not (self.nonlinearity_alpha >= 0.0):
            raise ValueError(f"require: nonlinearity_alpha ({self.nonlinearity_alpha}) >= 0.0")
        if not (self.drift_decay_rate >= 0.0):
            raise ValueError(f"require: drift_decay_rate ({self.drift_decay_rate}) >= 0.0")
        if not (self.drift_t0 >= 0.0):
            raise ValueError(f"require: drift_t0 ({self.drift_t0}) >= 0.0")
        if self.prog_gamma is not None:
            if not (self.prog_gamma.theta > 0.0):
                raise ValueError(f"require: prog_gamma.theta ({self.prog_gamma.theta}) > 0.0")
            if not (self.prog_gamma.k_intercept > 0.0):
                raise ValueError(f"require: prog_gamma.k_intercept ({self.prog_gamma.k_intercept}) > 0.0")
        if self.read_telegraph is not None:
            if not (0.0 <= self.read_telegraph.p_high_state <= 1.0):
                raise ValueError(
                    f"require: 0.0 <= read_telegraph.p_high_state ({self.read_telegraph.p_high_state}) <= 1.0"
                )
            if not (self.read_telegraph.amplitude_mean >= 0.0 and self.read_telegraph.amplitude_std >= 0.0):
                raise ValueError(
                    f"require: read_telegraph.amplitude_mean ({self.read_telegraph.amplitude_mean}) >= 0.0 "
                    f"and read_telegraph.amplitude_std ({self.read_telegraph.amplitude_std}) >= 0.0"
                )
        if self.read_thermal is not None and not (self.read_thermal >= 0.0):
            raise ValueError(f"require: read_thermal ({self.read_thermal}) >= 0.0")
        if self.stuck_at is not None:
            self.stuck_at.validate()
        if not (self.c_top__fF >= 0.0):
            raise ValueError(f"require: c_top__fF ({self.c_top__fF}) >= 0.0")
        if not (self.c_bot__fF >= 0.0):
            raise ValueError(f"require: c_bot__fF ({self.c_bot__fF}) >= 0.0")

    def __post_init__(self) -> None:
        self.validate()


@dataclass(frozen=True)
class RRAMDC:
    """Solver-facing working-point result for one RRAM cell evaluation.

    Returned by :meth:`RRAM.solve_dc`; frozen so ``@torch.compile``'s
    Dynamo tracer treats it as an immutable value rather than an
    object whose attributes might mutate.

    Attributes:
        i__uA: Device current ``I(v)`` [uA] through the cell at the
            requested operating point.
        di_dv__uS: Differential conductance ``∂I/∂V`` [uS] — the
            local linearisation a Newton solver needs for its
            Jacobian stamp at this cell.
    """

    i__uA: Tensor
    di_dv__uS: Tensor


@dataclass(frozen=True)
class RRAMSnapshot:
    """Per-VMM read-noise conductance snapshot.

    A frozen dataclass returned by :meth:`RRAM.snapshot` and
    consumed opaquely by :meth:`RRAM.solve_dc` and the surrounding
    1T1R solver.  Its lifetime is exactly one VMM — never registered
    as a buffer, never re-sampled inside the Newton iteration.

    Attributes:
        state_g__uS: Per-cell read conductance [uS] after RTN +
            thermal noise sampling, broadcast to the execution shape.
    """

    state_g__uS: Tensor


class RRAM(nn.Module):
    """Per-instance RRAM device model.

    Each :class:`RRAM` instance internally owns the post-program
    conductance state of one fabricated array — *not* a project-wide
    singleton.  Lifecycle:

    1. :meth:`program(state, t_elapsed=0.0)` programs the array:
       runs state→g lookup + Gamma + drift + stuck-at + clamp, and
       registers the result as ``self.state_g__uS`` (non-persistent).
       Re-callable: a subsequent :meth:`program` replaces the buffer.
    2. :meth:`snapshot(shape=...)` returns a fresh
       :class:`RRAMSnapshot` per VMM call — RTN + thermal noise
       are sampled per element on the execution shape, never persisted.
    3. :meth:`solve_dc(v, runtime)` evaluates the sinh I-V model and
       its linearisation using the runtime snapshot.

    The internal ``state_g__uS`` buffer is owned here so higher
    layers never have to thread the post-program tensor through
    function signatures.  Read noise is sampled once per VMM by
    :meth:`snapshot` and held by the caller as a local
    :class:`RRAMSnapshot`.
    """

    state_to_g_map__uS: Tensor
    state_g__uS: Tensor

    def __init__(
        self,
        cfg: RRAMConfig,
        *,
        name: str = "",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize RRAM device.

        Args:
            cfg: RRAM configuration parameters.
            name: Hierarchical profiler name forwarded by the parent core.
                RRAM does not profile its own PPA (rolled into the
                xbar tile's monolithic area in the config), so the name
                is held for diagnostics but no events fire.
            dtype: Torch float dtype for the lookup table and the
                fabricated state buffer.
        """
        super().__init__()

        self._neurox_name = name
        self.cfg = cfg
        self.dtype = dtype
        self.g_min__uS = cfg.g_min__uS
        self.g_max__uS = cfg.g_max__uS

        self.register_buffer("state_to_g_map__uS", torch.tensor(cfg.state_to_g_map__uS, dtype=dtype), persistent=False)
        self.register_buffer("state_g__uS", torch.empty(0, dtype=dtype), persistent=False)

    @property
    def num_states(self) -> int:
        """Number of discrete conductance states."""
        return self.cfg.num_states

    @property
    def c_top__fF(self) -> float:
        """Top-electrode (BL-side) parasitic capacitance per cell [fF]."""
        return self.cfg.c_top__fF

    @property
    def c_bot__fF(self) -> float:
        """Bottom-electrode (Node-X-side) parasitic capacitance per cell [fF]."""
        return self.cfg.c_bot__fF

    def program(self, state: Tensor, t_elapsed: float = 0.0) -> None:
        """Program this RRAM array (stateful).

        Applies, in order: (1) ideal state→g lookup, (2) state-dependent
        Gamma programming variation, (3) power-law temporal drift,
        (4) stuck-at faults, (5) clamp to ``[g_min, g_max]``.  Noise
        stages whose sub-configs are ``None`` are skipped by the
        ``apply_*`` helpers.  The resulting per-cell conductance is
        registered as the non-persistent buffer ``self.state_g__uS``.

        Re-callable: a subsequent :meth:`program` replaces the buffer
        with a fresh sample.

        Args:
            state: Integer state index tensor, values in
                ``[0, num_states - 1]``.  Shape: arbitrary (typically
                the array's ``[phys_col_num, row_num]``).
            t_elapsed: Time elapsed since programming [s].  Drift is
                applied only when ``drift_decay_rate > 0`` and
                ``t_elapsed > drift_t0``.
        """
        state_g__uS = self.state_to_g_map__uS[state.long()]
        state_g__uS = apply_state_dependent_gamma(state_g__uS, self.cfg.prog_gamma)
        if self.cfg.drift_decay_rate > 0.0 and t_elapsed > self.cfg.drift_t0:
            drift_factor = (t_elapsed / self.cfg.drift_t0) ** (-self.cfg.drift_decay_rate)
            state_g__uS = state_g__uS * drift_factor

        state_g__uS = apply_stuck_at_fault(
            x=state_g__uS, config=self.cfg.stuck_at, min_val=self.g_min__uS, max_val=self.g_max__uS
        )

        state_g__uS = state_g__uS.clamp(self.g_min__uS, self.g_max__uS)

        self.register_buffer("state_g__uS", state_g__uS, persistent=False)

    def snapshot(self, *, shape: tuple[int, ...]) -> RRAMSnapshot:
        """Sample one VMM's read-noise snapshot at the execution shape.

        Expands the post-program ``state_g__uS`` to the requested
        execution layout, applies telegraph + thermal noise per
        element, clamps to ``[g_min, g_max]``, and returns the
        snapshot wrapped in a :class:`RRAMSnapshot`.  This is the
        canonical per-VMM entry point — the solver and surrounding
        ``Core1T1R.forward`` call it once per VMM and reuse the
        returned dataclass for the entire iteration (per
        ``temp/state_holding.md``).

        ``apply_*`` helpers skip their stage when the corresponding
        sub-config is ``None``.  Noise is sampled **per element on the
        execution shape**, so each batch / tile position gets an
        independent draw.

        Args:
            shape: Target execution shape for per-element sampling.
                Required — callers must hand over the layout
                explicitly so per-element noise lands on the right
                tensor positions.

        Returns:
            :class:`RRAMSnapshot` carrying ``g_read__uS`` at the
            execution shape.
        """
        g__uS = self.state_g__uS.expand(shape)
        g__uS = apply_telegraph_noise(g__uS, self.cfg.read_telegraph)
        if self.cfg.read_thermal is not None:
            g__uS = apply_gaussian(g__uS, self.cfg.read_thermal)
        g__uS = g__uS.clamp(self.g_min__uS, self.g_max__uS)
        return RRAMSnapshot(state_g__uS=g__uS)

    def solve_dc(self, v__V: Tensor, snapshot: RRAMSnapshot) -> RRAMDC:
        """Evaluate the sinh I-V model and its local linearisation in one call.

        For ``alpha == 0``:  ``I = g · v``,        ``∂I/∂V = g``.
        For ``alpha > 0``:   ``I = g · sinh(α·v) / α``,
                             ``∂I/∂V = g · cosh(α·v)``.

        ``sinh`` and ``cosh`` share the same argument ``α·v``;
        returning both quantities together lets ``@torch.compile``
        emit a single fused kernel for the cell-Newton step that
        consumes both.

        Args:
            v__V: Applied voltage [V].  Shape: arbitrary
                (broadcast-compatible with ``snapshot.g_read__uS``).
            snapshot: Per-VMM read-noise snapshot produced by
                :meth:`snapshot`.

        Returns:
            :class:`RRAMDC` holding ``i__uA`` and ``di_dv__uS`` at
            the requested working point.  Both tensors broadcast to
            the shape of ``v__V · snapshot.g_read__uS``.
        """
        state_g__uS = snapshot.state_g__uS
        alpha = self.cfg.nonlinearity_alpha
        if alpha == 0.0:
            i__uA = state_g__uS * v__V
            di_dv__uS = state_g__uS.expand_as(i__uA)
        else:
            ax = alpha * v__V
            i__uA = state_g__uS * torch.sinh(ax) / alpha
            di_dv__uS = state_g__uS * torch.cosh(ax)
        return RRAMDC(i__uA=i__uA, di_dv__uS=di_dv__uS)
