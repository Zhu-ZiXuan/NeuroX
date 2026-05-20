"""Op-amp-based transimpedance amplifier used as a BL clamp driver.

See also:
    docs/dev/modules/analog/tia/opamp_tia.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.device.nmos import NMOS, NMOSConfig, NMOSSnapshot

from .base import TIA, TIAConfig


@dataclass(frozen=True, kw_only=True)
class OpAmpTIAConfig(TIAConfig):
    """Configuration for :class:`OpAmpTIA`.

    Attributes:
        v_nmos_bias__V: Pseudo-resistor gate bias [V].
        v_dd__V: Supply rail [V].
        opamp_gain: Nominal open-loop gain; must be > 1.
        opamp_gain_sigma: Relative mismatch (``σ/μ``) on ``opamp_gain``.
        enable_opamp_gain_sigma: Apply ``opamp_gain_sigma`` at fabricate
            time.
        nmos_cfg: PDK config for the pseudo-resistor NMOS.
        pseudo_nmos_W__um: Pseudo-resistor channel width [μm].
        pseudo_nmos_L__um: Pseudo-resistor channel length [μm].
        output_saturation_softness__V: Softness scale [V] for the
            ``tanh`` output-rail limiter.
    """

    # --- Bias / supply ---
    v_nmos_bias__V: float
    v_dd__V: float

    # --- Op-amp gain ---
    opamp_gain: float

    # --- Op-amp gain mismatch ---
    opamp_gain_sigma: float
    enable_opamp_gain_sigma: bool

    # --- Pseudo-resistor NMOS ---
    nmos_cfg: NMOSConfig
    pseudo_nmos_W__um: float
    pseudo_nmos_L__um: float

    # --- Output rail limiter ---
    output_saturation_softness__V: float

    def validate(self) -> None:
        super().validate()
        self.validate_opamp()
        self.validate_pseudo_nmos()

    def validate_opamp(self) -> None:
        if not (self.opamp_gain > 1.0):
            raise ValueError(f"require: opamp_gain ({self.opamp_gain}) > 1.0")
        self._require_nonneg(self.opamp_gain_sigma, "opamp_gain_sigma")
        self._require_pos(self.output_saturation_softness__V, "output_saturation_softness__V")

    def validate_pseudo_nmos(self) -> None:
        self._require_pos(self.pseudo_nmos_W__um, "pseudo_nmos_W__um")
        self._require_pos(self.pseudo_nmos_L__um, "pseudo_nmos_L__um")
        if not (self.v_dd__V > self.v_ref__V):
            raise ValueError(f"require: v_dd__V ({self.v_dd__V}) > v_ref__V ({self.v_ref__V})")
        if not (self.v_nmos_bias__V > self.v_ref__V):
            raise ValueError(f"require: v_nmos_bias__V ({self.v_nmos_bias__V}) > v_ref__V ({self.v_ref__V})")


@dataclass(frozen=True)
class OpAmpTIADCOP:
    """DC operating-point result of :meth:`OpAmpTIA.solve_dc`.

    Attributes:
        v_clamp__V: Clamp-node voltage at the converged operating point [V].
        v_out__V: Soft-saturated op-amp output voltage [V].
        dVclamp_dI__MOhm: ``∂v_clamp / ∂i_port`` [MOhm].
        dVout_dI__MOhm: ``∂v_out / ∂i_port`` [MOhm].
    """

    v_clamp__V: Tensor
    v_out__V: Tensor
    dVclamp_dI__MOhm: Tensor
    dVout_dI__MOhm: Tensor


@dataclass(frozen=True)
class OpAmpTIASnapshot:
    """Per-call OpAmpTIA snapshot.

    Attributes:
        opamp_gain: Sampled per-instance open-loop gain (unit-less).
        nmos_snapshot: Pseudo-resistor NMOS snapshot.
    """

    opamp_gain: Tensor
    nmos_snapshot: NMOSSnapshot


@TIA.register_key(OpAmpTIAConfig)
class OpAmpTIA(TIA):
    """Non-linear OpAmpTIA clamp driver."""

    nominal_opamp_gain: Tensor
    opamp_gain: Tensor

    def __init__(
        self,
        *,
        cfg: OpAmpTIAConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(cfg=cfg, name=name, T__K=T__K, dtype=dtype)

        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype

        self.nmos = NMOS(
            cfg=cfg.nmos_cfg,
            T__K=T__K,
            dtype=dtype,
            W__um=cfg.pseudo_nmos_W__um,
            L__um=cfg.pseudo_nmos_L__um,
        )

        self.sigma_opamp_gain = cfg.opamp_gain * cfg.opamp_gain_sigma

        self.softclip_center__V = cfg.v_dd__V / 2.0
        self.softclip_half_span__V = cfg.v_dd__V / 2.0
        self.softclip_softness__V = cfg.output_saturation_softness__V

        self.register_buffer(
            "nominal_opamp_gain",
            torch.tensor(cfg.opamp_gain, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "opamp_gain",
            self.nominal_opamp_gain.clone(),
            persistent=False,
        )

    # --- ClampDriver protocol accessor ---

    @property
    def v_ref__V(self) -> float:
        """Ideal reference clamp voltage [V]."""
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
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        self.nmos.fabricate(shape)

        opamp_gain = apply_gaussian(
            self.nominal_opamp_gain.clone().expand(shape),
            self.sigma_opamp_gain,
            enabled=self.cfg.enable_opamp_gain_sigma,
        )
        self.register_buffer("opamp_gain", opamp_gain, persistent=False)
        self._record_inst_count(shape)

    # --- snapshot ---

    def snapshot(self, *, shape: tuple[int, ...]) -> OpAmpTIASnapshot:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Snapshot shape.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        nmos_snapshot = self.nmos.snapshot(shape=shape)
        return OpAmpTIASnapshot(opamp_gain=self.opamp_gain, nmos_snapshot=nmos_snapshot)

    # --- forward path ---

    N_NEWTON: int = 4
    G_EFF_MAX__uS: float = -1e-6

    def _softclip_eval(self, v_out_lin__V: Tensor) -> tuple[Tensor, Tensor]:
        """Smooth output-rail limiter ``c + h · tanh((x - c) / s)`` and its derivative.

        Args:
            v_out_lin__V: Pre-clip op-amp output [V].

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
        snapshot: OpAmpTIASnapshot,
        *,
        v_clamp_init__V: Tensor | None = None,
    ) -> OpAmpTIADCOP:
        """Solve the closed-loop OpAmpTIA at one port current.

        Args:
            i_port__uA: Port-output current [uA]; positive = sourcing.
            snapshot: Per-call snapshot from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start; ``None`` seeds from the
                zero-current static op.

        Returns:
            Converged ``v_clamp``, soft-saturated ``v_out``, and the
            small-signal sensitivities ``∂v_clamp/∂i_port`` and
            ``∂v_out/∂i_port`` ([V/uA] = [MOhm]).
        """
        v_ref = self.cfg.v_ref__V
        v_nmos_bias = self.cfg.v_nmos_bias__V
        v_dd = self.cfg.v_dd__V

        opamp_gain = snapshot.opamp_gain
        nmos_snapshot = snapshot.nmos_snapshot

        # warm start: zero-current static op when none provided.
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
            # df/dVclamp = ∂I/∂v_d · (-A · g_clip) + ∂I/∂v_s
            dvout_dvclamp = -opamp_gain * g_clip
            df_dVclamp = (nmos_dc.did_dvd__uS * dvout_dvclamp + nmos_dc.did_dvs__uS).clamp(max=self.G_EFF_MAX__uS)
            residual = nmos_dc.ids__uA - i_port__uA
            # project into [0, v_dd] so the update stalls at the rail when
            # the residual has no zero in-range.
            v_clamp = (v_clamp - residual / df_dVclamp).clamp(min=0.0, max=v_dd)

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
        # implicit-function theorem: dVclamp/dI = 1 / df_dVclamp [MOhm]
        dVclamp_dI__MOhm = 1.0 / df_dVclamp_final
        dVout_dI__MOhm = dvout_dvclamp * dVclamp_dI__MOhm

        return OpAmpTIADCOP(
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
        """`ClampDriver`-protocol wrapper around :meth:`solve_dc`.

        Args:
            i_port__uA: Port-output current [uA]; see :meth:`solve_dc`.
            snapshot: Per-call snapshot from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start for the inner Newton.

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)``.
        """
        dc = self.solve_dc(i_port__uA, snapshot, v_clamp_init__V=v_clamp_init__V)
        return dc.v_clamp__V, dc.dVclamp_dI__MOhm
