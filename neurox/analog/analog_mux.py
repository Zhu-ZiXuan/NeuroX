"""Analog multiplexer — differential voltage-transport behavioural block.

Position in the signal chain::

    core_1t1r  ->  readout  ->  analog_mux  ->  differential ADC

``AnalogMux`` is a small behavioural block: it accepts a differential
pair ``(v_pos, v_neg)`` from the readout, applies a scalar transport
gain and an optional dynamic noise drawn fresh per call, and emits the
transported pair together with the per-access dynamic energy.

The class deliberately models **transport behaviour only** — it does
not solve an operating point, hold runtime state, or propagate timing
through an RC tree.  In particular it does **not** carry an internal
``snapshot`` lifecycle: dynamic noise is drawn inline inside
:meth:`select` via :func:`apply_gaussian`, matching the
:class:`~neurox.analog.switch_cap.SwitchCap` convention (one fresh
draw per kernel call, no per-VMM snapshot threaded from upstream).

Differential noise model
------------------------
Transport noise is split into a common-mode (CM) and a differential-mode
(DM) component, sampled independently per output element and combined
as::

    n_cm ~ N(0, σ_cm)   if mux_noise_cm_sigma__V is not None else 0
    n_dm ~ N(0, σ_dm)   if mux_noise_dm_sigma__V is not None else 0
    n_pos = n_cm + n_dm
    n_neg = n_cm - n_dm

A pure differential ADC suppresses the CM term, so this split lets a
chip-level study cleanly separate "noise the ADC rejects" from
"noise that actually hurts the signal".  Setting both sigmas to
``None`` (the default for a pass-through study) disables the noise
entirely.

Transport model
---------------
Per output element::

    v_pos_out = mux_gain * v_pos_in + n_pos
    v_neg_out = mux_gain * v_neg_in + n_neg

``mux_gain == 1.0`` together with both noise sigmas ``None`` makes the
MUX a pure pass-through; the per-access dynamic energy still accrues
so chip-level PPA accounting stays consistent.

Dynamic energy
--------------
Per access the MUX dissipates a fixed scalar ``energy_per_access__fJ``;
:meth:`select` broadcasts it to ``v_pos__V.shape`` so the macro's
element-wise reductions work uniformly.  Latency, leakage and area
are plain configured constants (``latency_per_op__ns``,
``leakage_per_inst__uW``, ``area_per_inst__um2``) — no pseudo-physical
RC derivation here, since the project does not track signal timing at
this granularity.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.profiler import ProfiledModule


@dataclass(frozen=True, kw_only=True)
class AnalogMuxConfig:
    """Immutable configuration for :class:`AnalogMux`.

    Attributes:
        energy_per_access__fJ: Per-access dynamic energy [fJ].  Plain
            configured constant — no internal pseudo-physical RC
            derivation.  Must be ``>= 0``.
        mux_gain: Scalar transport gain applied identically to both
            legs.  ``1.0`` is the ideal pass-through; values ``< 1``
            model settling-related attenuation.  Must be ``> 0``.
        mux_noise_cm_sigma__V: Optional common-mode dynamic noise
            sigma [V].  ``None`` disables CM noise.  The CM draw is
            added with the same sign to both legs and is suppressed
            by a differential ADC.
        mux_noise_dm_sigma__V: Optional differential-mode dynamic
            noise sigma [V].  ``None`` disables DM noise.  The DM
            draw is added to ``v_pos`` and subtracted from ``v_neg``,
            so it survives a differential ADC and directly hurts the
            signal.
        leakage_per_inst__uW: Static leakage per instance [uW].
        area_per_inst__um2: Silicon area per instance [um^2].
        latency_per_op__ns: Per-access latency [ns] — plain
            configured constant.
    """

    energy_per_access__fJ: float
    mux_gain: float
    mux_noise_cm_sigma__V: float | None
    mux_noise_dm_sigma__V: float | None
    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        if self.energy_per_access__fJ < 0.0:
            raise ValueError(f"AnalogMuxConfig.energy_per_access__fJ ({self.energy_per_access__fJ}) must be >= 0")
        if not (self.mux_gain > 0.0):
            raise ValueError(f"AnalogMuxConfig.mux_gain ({self.mux_gain}) must be > 0")
        if self.mux_noise_cm_sigma__V is not None and self.mux_noise_cm_sigma__V < 0.0:
            raise ValueError(
                f"AnalogMuxConfig.mux_noise_cm_sigma__V ({self.mux_noise_cm_sigma__V}) must be >= 0 or None"
            )
        if self.mux_noise_dm_sigma__V is not None and self.mux_noise_dm_sigma__V < 0.0:
            raise ValueError(
                f"AnalogMuxConfig.mux_noise_dm_sigma__V ({self.mux_noise_dm_sigma__V}) must be >= 0 or None"
            )
        if self.leakage_per_inst__uW < 0.0:
            raise ValueError(f"AnalogMuxConfig.leakage_per_inst__uW ({self.leakage_per_inst__uW}) must be >= 0")
        if self.area_per_inst__um2 < 0.0:
            raise ValueError(f"AnalogMuxConfig.area_per_inst__um2 ({self.area_per_inst__um2}) must be >= 0")
        if self.latency_per_op__ns < 0.0:
            raise ValueError(f"AnalogMuxConfig.latency_per_op__ns ({self.latency_per_op__ns}) must be >= 0")


class AnalogMux(nn.Module, ProfiledModule):
    """Differential voltage-transport block — gain + CM/DM noise + access energy.

    The MUX owns no static fabricated state (no ``fabricate``) and no
    per-VMM runtime snapshot.  Dynamic noise is drawn fresh inside
    :meth:`select` for each call, consistent with the
    :class:`~neurox.analog.switch_cap.SwitchCap` convention.

    Args:
        cfg: Immutable :class:`AnalogMuxConfig`.
        name: Hierarchical profiler name (e.g. ``<readout>.analog_mux``).
        dtype: Floating-point dtype for the energy tensor and the
            inline noise draws.
    """

    def __init__(self, cfg: AnalogMuxConfig, *, name: str = "", dtype: torch.dtype = torch.float32) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self.cfg = cfg

    @property
    def area_per_inst__um2(self) -> float:
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        return self.cfg.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """No-op — :class:`AnalogMux` owns no fabricated static state today.

        Kept on the API surface (and required to take ``shape``) to
        match the lifecycle convention of the other readout-chain
        modules (``SwitchCap``, ``ReadOut``, ``Core1T1R``).  A future
        iteration may introduce static mismatch (per-leg gain, CM/DM
        offset) sized over this ``shape``; until then it is a
        deliberate no-op.

        Args:
            shape: Physical-instance prefix shape ``*P`` plus the
                per-array transport-channel axis ``N`` — typically
                ``(*P, N)``.  Ignored today.
        """
        return

    def transport(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply gain + CM/DM transport noise to a differential pair.

        Inline (no snapshot threaded): the CM and DM noise tensors are
        drawn fresh each call via :func:`apply_gaussian` on zero
        tensors shaped like ``v_pos__V``.  When a sigma is ``None`` the
        corresponding draw is skipped.  Dynamic energy is emitted as
        a profiler side-channel event.

        Args:
            v_pos__V: Positive-leg input voltage [V].  Any shape.
            v_neg__V: Negative-leg input voltage [V].  Broadcastable
                with ``v_pos__V``.

        Returns:
            ``(v_pos_muxed__V, v_neg_muxed__V)`` — both tensors share
            ``v_pos__V``'s shape.
        """
        gain = self.cfg.mux_gain
        v_pos_muxed__V = gain * v_pos__V
        v_neg_muxed__V = gain * v_neg__V

        # CM/DM dynamic noise.  CM is added with matching sign to both
        # legs (suppressed by a differential ADC); DM is added to pos
        # and subtracted from neg (survives the differential ADC).
        sigma_cm__V = self.cfg.mux_noise_cm_sigma__V
        if sigma_cm__V is not None and sigma_cm__V > 0.0:
            zeros = torch.zeros_like(v_pos_muxed__V)
            n_cm__V = apply_gaussian(zeros, sigma_cm__V)
            v_pos_muxed__V = v_pos_muxed__V + n_cm__V
            v_neg_muxed__V = v_neg_muxed__V + n_cm__V
        sigma_dm__V = self.cfg.mux_noise_dm_sigma__V
        if sigma_dm__V is not None and sigma_dm__V > 0.0:
            zeros = torch.zeros_like(v_pos_muxed__V)
            n_dm__V = apply_gaussian(zeros, sigma_dm__V)
            v_pos_muxed__V = v_pos_muxed__V + n_dm__V
            v_neg_muxed__V = v_neg_muxed__V - n_dm__V

        dynamic_energy__fJ = torch.full_like(v_pos__V, self.cfg.energy_per_access__fJ)
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return v_pos_muxed__V, v_neg_muxed__V
