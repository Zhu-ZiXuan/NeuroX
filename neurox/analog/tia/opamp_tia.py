"""Op-amp-based transimpedance amplifier used as a BL clamp driver.

See also:
    docs/reference/analog/tia/opamp_tia.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.device import NMOS, MOSFETConfig, MOSFETPolicy, MOSFETSnap

from .base import TIA, TIAConfig, TIAPolicy, TIASnap


@dataclass(frozen=True, kw_only=True)
class OpAmpTIAConfig(TIAConfig):
    """Configuration for :class:`OpAmpTIA`.

    Attributes:
        v_nmos_bias__V: Pseudo-resistor gate bias.
        v_dd__V: Supply rail.
        opamp_gain: Nominal open-loop gain; must be > 1.
        opamp_gain_sigma: Relative mismatch (``σ/μ``) on ``opamp_gain``.
        nmos_config: PDK config for the pseudo-resistor NMOS.
        pseudo_nmos_W__um: Pseudo-resistor channel width.
        pseudo_nmos_L__um: Pseudo-resistor channel length.
        output_saturation_softness__V: Softness scale for the
            ``tanh`` output-rail limiter.
        n_newton: Step-damped Newton iterations in the closed-loop solve.
            Compile-time constant; pick via
            ``solver_calibrate.tia`` against the target
            workload.

    The Newton damping cap and the Jacobian-floor safety clamp are
    method-intrinsic constants on :class:`OpAmpTIA` (``MAX_STEP__V`` /
    ``G_EFF_MAX__uS``) — they are not exposed in this config because they
    do not vary per chip preset.
    """

    # --- Bias / supply ---
    v_nmos_bias__V: float
    v_dd__V: float

    # --- Op-amp gain ---
    opamp_gain: float

    # --- Op-amp gain mismatch ---
    opamp_gain_sigma: float

    # --- Pseudo-resistor NMOS ---
    nmos_config: MOSFETConfig
    pseudo_nmos_W__um: float
    pseudo_nmos_L__um: float

    # --- Output rail limiter ---
    output_saturation_softness__V: float

    # --- Newton solver iteration count ---
    n_newton: int

    def validate(self) -> None:
        super().validate()
        self.validate_opamp()
        self.validate_pseudo_nmos()
        self.validate_solver()

    def validate_opamp(self) -> None:
        if not (self.opamp_gain > 1.0):
            raise ValueError(f"require: opamp_gain ({self.opamp_gain}) > 1.0")
        self._require_non_neg(self.opamp_gain_sigma, "opamp_gain_sigma")
        self._require_pos(self.output_saturation_softness__V, "output_saturation_softness__V")

    def validate_pseudo_nmos(self) -> None:
        self._require_pos(self.pseudo_nmos_W__um, "pseudo_nmos_W__um")
        self._require_pos(self.pseudo_nmos_L__um, "pseudo_nmos_L__um")

    def validate_solver(self) -> None:
        self._require_pos(self.n_newton, "n_newton")


@dataclass(frozen=True)
class OpAmpTIAPolicy(TIAPolicy):
    """Per-source nonideality toggles for OpAmpTIA.

    Attributes:
        opamp_gain_sigma: Apply ``opamp_gain_sigma`` at fabricate time.
        nmos: Pseudo-resistor NMOS nonideality policy.
    """

    opamp_gain_sigma: bool
    nmos: MOSFETPolicy


@dataclass(frozen=True)
class OpAmpTIADCOP:
    """DC operating-point result of :meth:`OpAmpTIA.solve_dc`.

    Attributes:
        v_clamp__V: Clamp-node voltage at the converged operating point.
        v_out__V: Soft-saturated op-amp output voltage.
        dVclamp_dI__MOhm: ``∂v_clamp / ∂i_port``.
        dVout_dI__MOhm: ``∂v_out / ∂i_port``.
        residual__uA: ``|I_nmos(v_clamp) - i_port|`` at the final
            iterate. Consumed by
            ``solver_calibrate.tia`` to drive plateau
            detection over ``n_newton``.
    """

    v_clamp__V: Tensor
    v_out__V: Tensor
    dVclamp_dI__MOhm: Tensor
    dVout_dI__MOhm: Tensor
    residual__uA: Tensor


@dataclass(frozen=True)
class OpAmpTIASnap(TIASnap):
    """Per-call OpAmpTIA snap.

    Attributes:
        v_ref__V: Injected reference clamp voltage, broadcast to the
            per-call shape.
        opamp_gain: Sampled per-instance open-loop gain (unit-less).
        nmos_snap: Pseudo-resistor NMOS snap.
    """

    v_ref__V: Tensor
    opamp_gain: Tensor
    nmos_snap: MOSFETSnap


@TIA.register_key(OpAmpTIAConfig)
class OpAmpTIA(TIA[OpAmpTIASnap]):
    """Non-linear OpAmpTIA clamp driver.

    Class-level numerical constants (method-intrinsic, not chip-tuneable):

      * ``MAX_STEP__V``: Per-iteration ``|Δv_clamp|`` cap. Stops the
        damped Newton from sticking at a rail after a single overshoot in
        regions where ``tanh`` saturates (``g_clip → 0``).
      * ``G_EFF_MAX__uS``: Upper clamp on the effective KCL Jacobian
        ``df/dV_clamp``. Always negative; without it the Newton step
        diverges when the NMOS feedback loop's local derivative crosses
        zero.
    """

    MAX_STEP__V: float = 0.05
    G_EFF_MAX__uS: float = -1e-6

    config: OpAmpTIAConfig
    nominal_opamp_gain: Tensor
    opamp_gain: Tensor

    def __init__(
        self,
        *,
        config: OpAmpTIAConfig,
        policy: OpAmpTIAPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype

        self.nmos = NMOS(
            config=config.nmos_config,
            policy=policy.nmos,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
            W__um=config.pseudo_nmos_W__um,
            L__um=config.pseudo_nmos_L__um,
        )

        self.sigma_opamp_gain = config.opamp_gain * config.opamp_gain_sigma

        self.softclip_center__V = config.v_dd__V / 2.0
        self.softclip_half_span__V = config.v_dd__V / 2.0
        self.softclip_softness__V = config.output_saturation_softness__V

        self.register_buffer(
            "nominal_opamp_gain",
            torch.tensor(config.opamp_gain, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "opamp_gain",
            self.nominal_opamp_gain.clone(),
            persistent=False,
        )

    # --- fabricate ---

    def _sample_fabricate_mismatch(self) -> None:
        """Resample opamp_gain at ``self._inst_shape``."""
        self.opamp_gain = apply_gaussian(
            self.nominal_opamp_gain.clone().expand(self._inst_shape),
            self.sigma_opamp_gain,
            enabled=self.policy.opamp_gain_sigma,
        )

    # --- snapshot ---

    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> OpAmpTIASnap:
        """Sample one per-call runtime snap over ``shape``.

        Args:
            v_ref__V: Injected reference clamp voltage. A scalar or
                instance-shaped tensor that broadcasts onto ``shape``;
                stored in the returned snap.
            shape: Per-call broadcast shape; the snap fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state.
        """
        v_view = v_ref__V.expand(shape) if shape else v_ref__V
        v = v_view if multi_coords is None else v_view[multi_coords]
        gain_view = self.opamp_gain.expand(shape) if shape else self.opamp_gain
        gain = gain_view if multi_coords is None else gain_view[multi_coords]
        nmos_snap = self.nmos.snapshot(shape=shape, multi_coords=multi_coords)
        return OpAmpTIASnap(v_ref__V=v, opamp_gain=gain, nmos_snap=nmos_snap)

    # --- forward path ---

    def _softclip_eval(self, v_out_lin__V: Tensor) -> tuple[Tensor, Tensor]:
        """Smooth output-rail limiter and its derivative.

        Args:
            v_out_lin__V: Pre-clip op-amp output.

        Returns:
            ``(v_out, g_clip)`` — saturated output and local gradient.
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
        snap: OpAmpTIASnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> OpAmpTIADCOP:
        """Solve the closed-loop OpAmpTIA at one port current.

        Args:
            i_port__uA: Port-output current; positive = sourcing.
            snap: Per-call snap from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start; ``None`` seeds from the
                zero-current static op.

        Returns:
            Converged ``v_clamp``, soft-saturated ``v_out``, and the
            small-signal sensitivities ``∂v_clamp/∂i_port`` and
            ``∂v_out/∂i_port`` ([V/uA] = [MOhm]).
        """
        v_ref = snap.v_ref__V
        v_nmos_bias = self.config.v_nmos_bias__V
        v_dd = self.config.v_dd__V

        opamp_gain = snap.opamp_gain
        nmos_snap = snap.nmos_snap

        # warm start: zero-current static op when none provided.
        v_clamp = v_ref * opamp_gain / (opamp_gain + 1.0) if v_clamp_init__V is None else v_clamp_init__V

        max_step__V = self.MAX_STEP__V
        g_eff_max__uS = self.G_EFF_MAX__uS
        for _ in range(self.config.n_newton):
            v_out_lin = opamp_gain * (v_ref - v_clamp)
            v_out, g_clip = self._softclip_eval(v_out_lin)
            nmos_dc = self.nmos.solve_dc(
                vg__V=v_nmos_bias,
                vd__V=v_out,
                vs__V=v_clamp,
                snap=nmos_snap,
            )
            # clamp-loop KCL-residual Jacobian df/dV_clamp.
            dvout_dvclamp = -opamp_gain * g_clip
            df_dVclamp = (nmos_dc.did_dvd__uS * dvout_dvclamp + nmos_dc.did_dvs__uS).clamp(max=g_eff_max__uS)
            residual = nmos_dc.ids__uA - i_port__uA
            # Damped Newton step + physical projection. Without the per-step
            # cap, an overshoot near rail (where tanh's g_clip → 0) sends
            # v_clamp to the rail in one shot and the next iteration computes
            # f/f' at the rail — typically trapping the solver there.
            delta = (-residual / df_dVclamp).clamp(min=-max_step__V, max=max_step__V)
            v_clamp = (v_clamp + delta).clamp(min=0.0, max=v_dd)

        v_out_lin_final = opamp_gain * (v_ref - v_clamp)
        v_out, g_clip = self._softclip_eval(v_out_lin_final)
        nmos_dc_final = self.nmos.solve_dc(
            vg__V=v_nmos_bias,
            vd__V=v_out,
            vs__V=v_clamp,
            snap=nmos_snap,
        )
        dvout_dvclamp = -opamp_gain * g_clip
        df_dVclamp_final = (nmos_dc_final.did_dvd__uS * dvout_dvclamp + nmos_dc_final.did_dvs__uS).clamp(
            max=g_eff_max__uS
        )
        # implicit-function theorem: dVclamp/dI = 1 / df_dVclamp [MOhm]
        dVclamp_dI__MOhm = 1.0 / df_dVclamp_final
        dVout_dI__MOhm = dvout_dvclamp * dVclamp_dI__MOhm

        # Final-iterate KCL residual ``|NMOS Ids - i_port|`` returned for
        # ``solver_calibrate.tia`` to consume.
        residual__uA = (nmos_dc_final.ids__uA - i_port__uA).abs()

        return OpAmpTIADCOP(
            v_clamp__V=v_clamp,
            v_out__V=v_out,
            dVclamp_dI__MOhm=dVclamp_dI__MOhm,
            dVout_dI__MOhm=dVout_dI__MOhm,
            residual__uA=residual__uA,
        )

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: OpAmpTIASnap,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Boundary-clamp wrapper around :meth:`solve_dc`.

        Args:
            i_port__uA: Port-output current; see :meth:`solve_dc`.
            snap: Per-call snap from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start for the inner Newton.

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``.
        """
        dc = self.solve_dc(i_port__uA, snap, v_clamp_init__V=v_clamp_init__V)
        return dc.v_clamp__V, dc.dVclamp_dI__MOhm
