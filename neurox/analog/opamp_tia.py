"""Op-amp-based transimpedance amplifier — non-linear BL clamp driver.

:class:`OpAmpTIA` models the op-amp + NMOS-pseudo-resistor feedback
topology used as a BL clamp in 1T1R crossbar tiles.  It satisfies the
:class:`~neurox.analog.ClampDriver` Protocol — exposes
``v_ref__V``, :meth:`fabricate`, :meth:`snapshot`, and
:meth:`solve_clamp`.  The Protocol's :meth:`solve_clamp` is the
solver-facing surface returning a plain ``tuple[Tensor, Tensor]``;
:meth:`solve_dc` is the richer concrete-class entry point returning
:class:`OpAmpTIADC` for higher-layer consumers (readout path, energy
aggregation) that need OpAmpTIA-specific extras like ``v_out__V``.
The 1T1R DC solver consumes the boundary structurally through the
Protocol — without importing :class:`OpAmpTIA` directly.

Physical model
--------------
1. An op-amp pins the BL virtual-ground node near a reference voltage
   ``v_ref`` via negative feedback with finite open-loop gain ``A``.
   The actual BL clamp drifts as ``v_clamp ≈ v_ref − v_out / A``.
2. The feedback element is an NMOS pseudo-resistor biased in deep
   triode by a fixed gate voltage ``v_nmos_bias`` (typically VDD).
   Its drain sits at ``v_out``; its source sits at ``v_clamp``.
   The transistor sources whatever current the upstream load demands
   — within its linear-region capacity.
3. As ``i_in`` grows, ``v_ds = v_out − v_clamp`` grows, the
   pseudo-resistor's effective resistance climbs non-linearly, and
   the op-amp's output stage eventually runs into the supply rail.
   The rail limit is modelled by a **smooth softclip on
   ``v_out_lin = A · (v_ref − v_clamp)``** placed *inside* the
   residual Newton solves — saturation is part of the equation, not
   a post-processing step (see ``temp/opamp_tia.md``).

API surface
-----------
* :attr:`v_ref__V` — read-only property exposing the configured
  reference voltage.  The 1T1R solver reads it once per :meth:`solve`
  call to seed the Padé warm start (no separate clamp-driver
  evaluation is needed before the first cell-current estimate).
* :meth:`fabricate` — sample per-column mismatch and register
  ``self.opamp_gain`` internally; delegate to the per-instance NMOS
  submodule so the pseudo-resistor's ``β`` / ``V_th`` are likewise
  fabricated in place.
* :meth:`snapshot` — return a :class:`OpAmpTIASnapshot` aggregating the
  OpAmpTIA NMOS runtime snapshot.  Empty fields today; reserved for
  future dynamic noise.
* :meth:`solve_dc` — 4-iter unrolled 1D Newton on the inverting-input
  voltage ``v_clamp`` with the rail-limited output baked into the
  residual; returns :class:`OpAmpTIADC` with the converged
  ``v_clamp``, the soft-saturated ``v_out``, and the small-signal
  sensitivities ``dVclamp_dI__MOhm`` and ``dVout_dI__MOhm`` — both
  rolling smoothly to zero in deep saturation rather than being
  hard-masked.  Higher-level callers that need ``v_out`` for the
  readout path re-invoke :meth:`solve_dc` once with the converged
  column current and read ``OpAmpTIADC.v_out__V`` off the result.
* :meth:`solve_clamp` — :class:`ClampDriver`-Protocol wrapper around
  :meth:`solve_dc`.  Runs the same physical solve and returns the
  pair ``(v_clamp__V, dVclamp_dI__MOhm)`` the BL Newton step needs.
  The wrapper exists so the structural Protocol can pin the
  solver-facing return type to a fixed ``tuple[Tensor, Tensor]``
  without leaking :class:`OpAmpTIADC` through the Protocol — mypy treats
  method-return-as-Protocol unstably across concrete dataclass
  returns of competing implementations (OpAmpTIA vs Driver vs future CSA),
  so the explicit tuple surface is preferred.

Solver coupling
---------------
``solve_dc`` runs an unrolled 4-iteration 1D Newton with an
**analytical derivative** of the rail-limited residual:

    v_out_lin  = A · (v_ref − v_clamp)
    v_out      = softclip(v_out_lin; 0, v_dd, s)
    g_clip     = d softclip / dv_out_lin
    f(v_clamp) = nmos.ids(v_g=v_nmos_bias, v_d=v_out, v_s=v_clamp) − i_in
    df/dVclamp = ∂I/∂v_d · (−A · g_clip) + ∂I/∂v_s

The same softclip is used for the final operating-point evaluation,
so the returned ``v_out``, ``v_clamp``, ``dVclamp_dI__MOhm`` and
``dVout_dI__MOhm`` all come from one self-consistent smooth model.
``dVout_dI = (−A · g_clip) · dVclamp_dI`` naturally rolls toward zero
as the output saturates (``g_clip → 0``); no piecewise mask is
applied.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.device.nmos import NMOS, NMOSSnapshot
from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class OpAmpTIAConfig:
    """Immutable configuration for :class:`OpAmpTIA`.

    Attributes:
        v_ref__V: Virtual-ground reference voltage [V].  In the
            zero-current limit ``v_clamp → v_ref · A / (A + 1)``
            (modulo a small softclip-induced offset; see
            :meth:`OpAmpTIA.solve_dc`).
        v_nmos_bias__V: NMOS pseudo-resistor gate bias [V].  Held at
            a fixed potential (typically VDD); the corresponding
            overdrive ``v_nmos_bias − v_clamp − vth`` sets the pseudo-
            resistor's deep-triode capacity.
        v_dd__V: Supply rail used as the upper bound of the smooth
            output saturation on ``v_out_lin``.  The lower rail is
            ground (0 V).
        opamp_gain: Nominal open-loop gain (unit-less).  Must be > 1.
        opamp_gain_sigma: Optional Pelgrom-style relative mismatch
            (``σ/μ``) on ``opamp_gain``.  ``None`` disables per-
            instance gain variation.
        output_saturation_softness__V: Optional softness scale [V] for
            the smooth ``tanh``-based output rail limiter.  Controls
            the transition width between the linear interior and the
            soft-saturated rails.  When ``None`` the default is the
            rail half-span ``v_dd / 2`` — the natural ``tanh`` form
            ``c + h · tanh((x − c) / h)`` with unit slope at the
            center.  Smaller values give a sharper transition;
            larger values give a gentler one.  Must be > 0 if set.
        leakage_per_inst__uW: Static leakage per OpAmpTIA instance [uW] —
            covers the op-amp's quiescent bias contribution.
        area_per_inst__um2: Silicon area per OpAmpTIA instance [um²].
        latency_per_op__ns: Settling latency per VMM [ns].
    """

    v_ref__V: float
    v_nmos_bias__V: float
    v_dd__V: float

    opamp_gain: float
    opamp_gain_sigma: float | None = None

    output_saturation_softness__V: float | None = None

    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0
    latency_per_op__ns: float = 0.0

    def validate(self) -> None:
        """Validate OpAmpTIA configuration parameters."""
        if not (self.opamp_gain > 1.0):
            raise ValueError(f"require: opamp_gain ({self.opamp_gain}) > 1.0")
        if self.opamp_gain_sigma is not None and not (self.opamp_gain_sigma >= 0.0):
            raise ValueError(f"require: opamp_gain_sigma ({self.opamp_gain_sigma}) >= 0.0")
        if not (self.v_dd__V > self.v_ref__V):
            raise ValueError(f"require: v_dd__V ({self.v_dd__V}) > v_ref__V ({self.v_ref__V})")
        if not (self.v_nmos_bias__V > self.v_ref__V):
            raise ValueError(
                f"require: v_nmos_bias__V ({self.v_nmos_bias__V}) > v_ref__V ({self.v_ref__V}) "
                "(pseudo-resistor must be in strong inversion at the static operating point)"
            )
        if self.output_saturation_softness__V is not None and not (self.output_saturation_softness__V > 0.0):
            raise ValueError(f"require: output_saturation_softness__V ({self.output_saturation_softness__V}) > 0.0")
        if not (self.leakage_per_inst__uW >= 0.0):
            raise ValueError(f"require: leakage_per_inst__uW ({self.leakage_per_inst__uW}) >= 0.0")
        if not (self.area_per_inst__um2 >= 0.0):
            raise ValueError(f"require: area_per_inst__um2 ({self.area_per_inst__um2}) >= 0.0")
        if not (self.latency_per_op__ns >= 0.0):
            raise ValueError(f"require: latency_per_op__ns ({self.latency_per_op__ns}) >= 0.0")

    def __post_init__(self) -> None:
        self.validate()


@dataclass(frozen=True)
class OpAmpTIADC:
    """Richer DC working-point result for one OpAmpTIA evaluation.

    Returned by :meth:`OpAmpTIA.solve_dc`; frozen so ``@torch.compile``'s
    Dynamo tracer treats it as an immutable value.  Carries the
    OpAmpTIA's full working-point state (``v_clamp`` / ``v_out``) plus
    both small-signal sensitivities (``dVclamp_dI`` / ``dVout_dI``).
    The solver-facing :meth:`OpAmpTIA.solve_clamp` projects the first two —
    ``v_clamp__V`` and ``dVclamp_dI__MOhm`` — into a plain
    ``tuple[Tensor, Tensor]`` matching the
    :class:`~neurox.analog.ClampDriver` Protocol.  Higher-level
    callers (energy accumulator, readout) that need the OpAmpTIA-specific
    extras read them directly off this dataclass.

    Attributes:
        v_clamp__V: Converged virtual-ground voltage [V] (the op-amp's
            inverting input).  Read by the solver as the BL clamp
            voltage at the boundary.  Related to ``v_out__V`` through
            the unclipped relation ``v_out_lin = A · (v_ref − v_clamp)``
            and the softclip ``v_out__V = softclip(v_out_lin)``.
        v_out__V: Converged op-amp output voltage [V] after the smooth
            rail limiter.  Stays inside ``(0, v_dd)`` and approaches
            either rail asymptotically as the linear pre-clip drive
            ``v_out_lin`` runs off the rail.  Consumed by the readout
            path.
        dVclamp_dI__MOhm: ``∂v_clamp/∂i_in`` [MOhm] — small-signal
            sensitivity of the virtual-ground voltage to the input
            current.  Negative; magnitude rolls smoothly toward
            ``1 / |∂I/∂v_s|`` as the output saturates (the op-amp
            feedback fades and the bare pseudo-resistor sets the
            input impedance).  Read by the solver as the BL boundary
            input impedance.
        dVout_dI__MOhm: ``∂v_out/∂i_in`` [MOhm] — small-signal
            transimpedance.  Positive; equals
            ``(−A · g_clip) · dVclamp_dI``.  Rolls smoothly toward
            zero as ``g_clip → 0`` in the saturated region.
    """

    v_clamp__V: Tensor
    v_out__V: Tensor
    dVclamp_dI__MOhm: Tensor
    dVout_dI__MOhm: Tensor


@dataclass(frozen=True)
class OpAmpTIASnapshot:
    """Per-VMM OpAmpTIA snapshot.

    Carries the per-VMM materialized device parameters consumed by
    :meth:`OpAmpTIA.solve_dc`.  Sampled once per VMM by
    :meth:`OpAmpTIA.snapshot` and threaded through both the solver's
    outer-iteration :meth:`OpAmpTIA.solve_dc` calls and the post-solver
    re-call in ``Core1T1R.forward`` so the Newton iteration and the
    readout-path ``v_out`` reconstruction share an identical fixed
    snapshot.

    Attributes:
        opamp_gain: Per-column op-amp gain [unit-less].
        nmos_snapshot: Nested OpAmpTIA pseudo-resistor NMOS snapshot.
    """

    opamp_gain: Tensor
    nmos_snapshot: NMOSSnapshot


class OpAmpTIA(nn.Module, ProfiledModule):
    """Non-linear OpAmpTIA — concrete BL clamp driver (per-instance).

    Each :class:`OpAmpTIA` owns its own pseudo-resistor NMOS submodule and
    its own per-column ``opamp_gain`` fabricated buffer.  No state is
    shared across OpAmpTIA instances.

    Buffers follow the project-wide **nominal vs. fabricated** split:

    * ``nominal_opamp_gain`` — 0-d scalar at ``cfg.opamp_gain``.
      Built once at ``__init__`` and never overwritten.
    * ``opamp_gain`` — fabricated per-column gain.

    :meth:`fabricate(shape)` rebuilds ``opamp_gain`` from
    ``nominal_opamp_gain`` on every call via ``clone().expand(shape)``
    and applies the optional Pelgrom-style relative Gaussian
    mismatch.  When ``opamp_gain_sigma`` is ``None`` the fabricated
    buffer stays a 1-element view (no full-shape memory).

    Static configuration scalars (``v_ref__V``, ``v_nmos_bias__V``,
    ``v_dd__V``) live on ``self.cfg`` as plain Python floats — *not*
    registered buffers — so ``torch.compile`` folds them as kernel
    constants.

    Args:
        cfg: Immutable OpAmpTIA configuration.
        nmos_factory: Zero-argument factory building this OpAmpTIA's
            internal pseudo-resistor NMOS submodule.  Each OpAmpTIA gets
            its own NMOS instance — different physical transistor
            from the array's access NMOS (separate W/L, separate
            bias point) and isolated from any other OpAmpTIA's mismatch
            sampling.
        dtype: Floating-point dtype used by the registered buffers.
    """

    nominal_opamp_gain: Tensor
    opamp_gain: Tensor

    def __init__(
        self,
        cfg: OpAmpTIAConfig,
        *,
        name: str = "",
        nmos_factory: Callable[..., NMOS],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

        self.cfg = cfg
        self.nmos: NMOS = nmos_factory(name=f"{name}.nmos" if name else "nmos")
        self.dtype = dtype

        self.sigma_opamp_gain: float | None = (
            cfg.opamp_gain * cfg.opamp_gain_sigma if cfg.opamp_gain_sigma is not None else None
        )

        # Smooth output-rail softclip scalars — folded as kernel
        # constants by ``torch.compile``.  Center / half-span are the
        # rail midpoint and half-span (lower rail = 0, upper rail =
        # v_dd).  Softness defaults to the half-span, recovering the
        # natural symmetric ``c + h · tanh((x − c) / h)`` form with
        # unit slope at the center.
        self.softclip_center__V: float = cfg.v_dd__V / 2.0
        self.softclip_half_span__V: float = cfg.v_dd__V / 2.0
        self.softclip_softness__V: float = (
            cfg.output_saturation_softness__V
            if cfg.output_saturation_softness__V is not None
            else self.softclip_half_span__V
        )

        self.register_buffer(
            "nominal_opamp_gain",
            torch.tensor(cfg.opamp_gain, dtype=dtype),
            persistent=False,
        )
        # Sentinel fabricated buffer — :meth:`fabricate` overwrites
        # it.  Initialised to a fresh clone of the nominal (no
        # expand) so ``.to(device)`` migrates cleanly even
        # pre-fabricate.
        self.register_buffer(
            "opamp_gain",
            self.nominal_opamp_gain.clone(),
            persistent=False,
        )

    # --- ClampDriver protocol accessor ---

    @property
    def v_ref__V(self) -> float:
        """Ideal reference clamp voltage [V] — solver-seed input.

        Exposed as a property (rather than just inheriting from
        ``self.cfg``) so :class:`OpAmpTIA` satisfies the
        :class:`~neurox.analog.ClampDriver` Protocol structurally.
        """
        return self.cfg.v_ref__V

    # --- PPA accessors ---

    @property
    def area_per_inst__um2(self) -> float:
        """Area per OpAmpTIA instance [um²]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per OpAmpTIA instance [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Settling latency per VMM [ns]."""
        return self.cfg.latency_per_op__ns

    # --- fabricate ---

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample per-column mismatch for opamp gain and pseudo-resistor.

        ``opamp_gain`` is rebuilt from ``nominal_opamp_gain`` via
        ``clone().expand(shape)``; the optional Pelgrom-style
        relative Gaussian mismatch is applied on top.  When
        ``sigma_opamp_gain`` is ``None`` the fabricated buffer stays
        a 1-element view (no full-shape memory).  The pseudo-resistor
        NMOS submodule's ``fabricate`` is delegated so its ``β`` /
        ``V_th`` buffers land internally inside ``self.nmos``.

        Re-callable: a subsequent :meth:`fabricate` always restarts
        from the unchanged ``nominal_opamp_gain``.

        Args:
            shape: Per-column shape (typically ``(physical_col_num,)``
                for the offset 1T1R xbar).
        """
        self.nmos.fabricate(shape)

        opamp_gain = self.nominal_opamp_gain.clone().expand(shape)
        if self.sigma_opamp_gain is not None:
            opamp_gain = apply_gaussian(opamp_gain, self.sigma_opamp_gain)
        self.register_buffer("opamp_gain", opamp_gain, persistent=False)

    # --- snapshot ---

    def snapshot(self, *, shape: tuple[int, ...]) -> OpAmpTIASnapshot:
        """Return a per-VMM OpAmpTIA snapshot.

        Wraps the fabricated ``self.opamp_gain`` and the nested
        :class:`NMOSSnapshot` from ``self.nmos.snapshot`` so the
        Newton iteration and the post-solver re-call (for
        ``v_out__V``) share an identical fixed sample of any future
        dynamic noise.

        Args:
            shape: Execution shape for runtime sampling.  Forwarded
                to ``self.nmos.snapshot``.

        Returns:
            :class:`OpAmpTIASnapshot` with ``opamp_gain`` and a nested
            :class:`NMOSSnapshot`.
        """
        nmos_snapshot = self.nmos.snapshot(shape=shape)
        return OpAmpTIASnapshot(opamp_gain=self.opamp_gain, nmos_snapshot=nmos_snapshot)

    # --- forward path (called by the 1T1R solver) ---

    # 4-iter unrolled 1D Newton with analytical derivative.  3 iters
    # would suffice on a monotone single-variable residual; we keep 4
    # as a one-iter safety margin while still fitting inside the
    # surrounding ``@torch.compile`` kernel without dynamic control
    # flow.
    N_NEWTON: int = 4
    G_EFF_MAX__uS: float = -1e-6

    def _softclip_eval(self, v_out_lin__V: Tensor) -> tuple[Tensor, Tensor]:
        """Smooth output-rail limiter and its derivative.

        Implements ``softclip(x) = c + h · tanh((x − c) / s)`` with
        ``c = v_dd / 2``, ``h = v_dd / 2``, ``s`` the configured
        softness (defaults to ``h``).  Returns ``(v_out, g_clip)``
        where ``g_clip = d softclip / dx``.  Both tensors broadcast to
        ``v_out_lin__V``'s shape.

        Args:
            v_out_lin__V: Pre-clip op-amp output ``A · (v_ref −
                v_clamp)`` [V].

        Returns:
            Two tensors: the soft-saturated output voltage and the
            local softclip gradient ``g_clip`` (unit-less, in ``[0,
            h/s]``).
        """
        c = self.softclip_center__V
        h = self.softclip_half_span__V
        s = self.softclip_softness__V
        tanh_val = torch.tanh((v_out_lin__V - c) / s)
        v_out = c + h * tanh_val
        g_clip = (h / s) * (1.0 - tanh_val * tanh_val)
        return v_out, g_clip

    def solve_dc(
        self,
        i_port__uA: Tensor,
        snapshot: OpAmpTIASnapshot,
        *,
        v_clamp_init__V: Tensor | None = None,
    ) -> OpAmpTIADC:
        """Solve the closed-loop OpAmpTIA equation for the steady-state voltages and sensitivities.

        Given the **port-output current** ``i_port__uA`` — positive
        when current leaves the clamp port (OpAmpTIA is sourcing into the
        external circuit), negative when it enters the port — compute
        the virtual-ground voltage ``v_clamp`` and op-amp output
        voltage ``v_out`` that satisfy

            v_out_lin  = opamp_gain · (v_ref − v_clamp)
            v_out      = softclip(v_out_lin; 0, v_dd, softness)
            i_port__uA = nmos.ids(v_g=v_nmos_bias, v_d=v_out,
                                  v_s=v_clamp)

        plus the small-signal sensitivities ``∂v_clamp/∂i_port`` and
        ``∂v_out/∂i_port``.  ``V / uA = MOhm`` by units, so the
        sensitivities carry an ``__MOhm`` suffix.  The softclip
        smoothly limits ``v_out`` to ``(0, v_dd)``; saturation is part
        of the solved residual, not a post-processing step.

        Sign convention check: with ``v_out > v_clamp`` (OpAmpTIA pulled
        high to keep v_clamp pinned), the NMOS pseudo-resistor
        conducts ``ids > 0`` from drain (v_out) to source (v_clamp);
        current then exits the v_clamp node toward the external load
        — exactly the port-output convention.  Hence
        ``nmos.ids = i_port__uA`` directly, with no sign flip.

        Reads ``opamp_gain`` from the per-VMM ``snapshot`` (not from
        ``self``) and threads the nested :class:`NMOSSnapshot` down
        to ``self.nmos.solve_dc`` so the Newton iteration and the
        post-solver re-call see an identical fixed sample of any
        future dynamic noise.

        Algorithm:

        * 4 unrolled Newton iterations on the residual
          ``f(v_clamp) = i_nmos − i_port`` with the analytical
          derivative

              df/dVclamp = ∂I/∂v_d · (−A · g_clip) + ∂I/∂v_s

          where ``g_clip = d softclip / dv_out_lin`` evaluated at
          ``v_out_lin = A · (v_ref − v_clamp)``.  One
          :meth:`NMOS.solve_dc` per iter at the soft-saturated
          ``v_d = v_out``.
        * One final ``solve_dc`` at the converged ``v_clamp`` to
          compute the returned sensitivities from the same equation,
          with ``dVout_dI = (−A · g_clip) · dVclamp_dI`` rolling
          smoothly to zero in saturation.

        Args:
            i_port__uA: Port-output current [uA] — positive = OpAmpTIA
                sourcing into the external load, negative = OpAmpTIA
                sinking from the load.
            snapshot: Per-VMM OpAmpTIA snapshot from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start for ``v_clamp``.
                When ``None`` (the default), the routine starts from
                the zero-current static operating point
                ``v_ref·A/(A+1)`` (the unclipped seed; the softclip
                shifts the true equilibrium by ``O(1/A)``, well
                within Newton's convergence basin).  When supplied —
                typically the previous outer-iteration's ``v_clamp``
                from the calling crossbar solver — the Newton
                contracts from a much closer seed.

        Returns:
            :class:`OpAmpTIADC` with the converged ``v_clamp`` and
            soft-saturated ``v_out`` and the two sensitivities
            ``dVclamp_dI``, ``dVout_dI`` — all derived from one
            self-consistent smooth model.
        """
        v_ref = self.cfg.v_ref__V
        v_nmos_bias = self.cfg.v_nmos_bias__V
        v_dd = self.cfg.v_dd__V

        opamp_gain = snapshot.opamp_gain
        nmos_snapshot = snapshot.nmos_snapshot

        # Warm start: caller-supplied estimate if given (typically the
        # previous outer iteration's ``v_clamp``), otherwise the
        # zero-current static op ``v_out = v_clamp`` intersected with
        # ``v_out = A · (v_ref − v_clamp)``.
        v_clamp = v_ref * opamp_gain / (opamp_gain + 1.0) if v_clamp_init__V is None else v_clamp_init__V

        for _ in range(self.N_NEWTON):
            v_out_lin = opamp_gain * (v_ref - v_clamp)
            v_out, g_clip = self._softclip_eval(v_out_lin)
            nmos_dc = self.nmos.solve_dc(
                vg__V=v_nmos_bias,
                vd__V=v_out,
                vs__V=v_clamp,
                snapshot=nmos_snapshot,
            )
            # df/dVclamp = ∂I/∂v_d · (dv_d/dVclamp) + ∂I/∂v_s · (dv_s/dVclamp)
            #            = ∂I/∂v_d · (−A · g_clip) + ∂I/∂v_s · (+1)
            dvout_dvclamp = -opamp_gain * g_clip
            df_dVclamp = (nmos_dc.did_dvd__uS * dvout_dvclamp + nmos_dc.did_dvs__uS).clamp(max=self.G_EFF_MAX__uS)
            residual = nmos_dc.ids__uA - i_port__uA
            # Project the Newton update back into the physical clamp
            # range ``[0, v_dd]`` — when the outer crossbar solver is
            # mid-convergence it can pass an ``i_port`` that exceeds the
            # NMOS pseudo-resistor's deliverable current and the
            # in-loop softclipped residual has no zero.  A bare Newton
            # step would extrapolate wildly; the projection stalls
            # ``v_clamp`` at the physical rail instead, letting the
            # outer Newton converge while we report a bounded boundary
            # value.
            v_clamp = (v_clamp - residual / df_dVclamp).clamp(min=0.0, max=v_dd)

        # Final eval at converged v_clamp — sensitivities computed
        # from the same softclip-aware residual so the returned
        # operating point and its derivatives are self-consistent.
        v_out_lin_final = opamp_gain * (v_ref - v_clamp)
        v_out, g_clip = self._softclip_eval(v_out_lin_final)
        nmos_dc_final = self.nmos.solve_dc(
            vg__V=v_nmos_bias,
            vd__V=v_out,
            vs__V=v_clamp,
            snapshot=nmos_snapshot,
        )
        dvout_dvclamp = -opamp_gain * g_clip
        df_dVclamp_final = (nmos_dc_final.did_dvd__uS * dvout_dvclamp + nmos_dc_final.did_dvs__uS).clamp(
            max=self.G_EFF_MAX__uS
        )
        # Implicit-function theorem on ``f(v_clamp; i_port) = 0``:
        #   dVclamp/dI = − (∂f/∂i_port) / (∂f/∂v_clamp)
        #              = 1 / df_dVclamp                    (negative)
        # dVout/dI    = (dv_out / dv_clamp) · dVclamp/dI
        #              = (−A · g_clip) · dVclamp/dI        (positive)
        # Units: 1 / uS = V / uA = MOhm.  ``g_clip → 0`` in saturation
        # naturally drives ``dVout/dI → 0`` without any masking.
        dVclamp_dI__MOhm = 1.0 / df_dVclamp_final
        dVout_dI__MOhm = dvout_dvclamp * dVclamp_dI__MOhm

        return OpAmpTIADC(
            v_clamp__V=v_clamp,
            v_out__V=v_out,
            dVclamp_dI__MOhm=dVclamp_dI__MOhm,
            dVout_dI__MOhm=dVout_dI__MOhm,
        )

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snapshot: OpAmpTIASnapshot,
        *,
        v_clamp_init__V: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Solver-facing :class:`ClampDriver` wrapper around :meth:`solve_dc`.

        Runs the same 4-iter Newton physical solve as :meth:`solve_dc`
        and projects the result onto the two tensors the BL Newton
        step actually consumes —
        ``(v_clamp__V, dVclamp_dI__MOhm)``.  No separate solve, no
        recomputation: the OpAmpTIA-specific extras carried by
        :class:`OpAmpTIADC` (``v_out__V`` / ``dVout_dI__MOhm``) are simply
        dropped at the wrapper boundary.  Higher-level callers
        (readout, energy aggregation) that need those extras call
        :meth:`solve_dc` directly on the concrete OpAmpTIA instance.

        Args:
            i_port__uA: Port-output current [uA]; see
                :meth:`solve_dc` for the sign convention.  Forwarded
                verbatim to :meth:`solve_dc`.
            snapshot: Per-VMM snapshot from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start for the inner
                ``v_clamp`` Newton.  Forwarded verbatim.

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``  matching the
            :class:`~neurox.analog.ClampDriver` Protocol.
        """
        dc = self.solve_dc(i_port__uA, snapshot, v_clamp_init__V=v_clamp_init__V)
        return dc.v_clamp__V, dc.dVclamp_dI__MOhm
