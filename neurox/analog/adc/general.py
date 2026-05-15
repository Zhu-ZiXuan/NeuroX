"""GeneralADC — boundary-bucketize ADC with arbitrary thresholds.

The simplest concrete ADC: a sorted list of comparator thresholds
plus three optional Gaussian-noise stages.  Useful for calibration
work (where the boundary list is fitted offline by
:mod:`neurox.tools.xbar_adc_boundaries`) and as a behavioural fallback
when no specific topology is being studied.

Single-mode runtime contract: ``convert(..., mode=0, bits=max_bits)``
is the only legal point; the runtime args are validated against the
fixed bucketize grid.

Floor semantics
---------------
``boundaries`` should be placed at code edges
(``B_C = C · LSB`` for ``C ∈ {1, …, N - 1}``); the calibration tool
emits boundaries in this convention.  Stochastic rounding (active in
training mode unless overridden) draws ``uniform(0, LSB)`` jitter
before bucketize and is unbiased.

The legacy ``drive(shape)`` method is preserved as a backward-
compatible shim.  New code should drive the BL clamp through
:class:`neurox.analog.opamp_tia.OpAmpTIA` and treat this class strictly as a
converter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.common.quant import floor_bucketize

from .base import ADC


@dataclass(frozen=True)
class GeneralADCConfig:
    """Immutable configuration for :class:`GeneralADC`.

    Attributes:
        boundaries: Sorted comparator thresholds in input units (uA
            for current-mode, V for voltage-mode), excluding the
            implicit ±∞ outer bounds.  ``N`` boundaries define
            ``N + 1`` output codes ``[0, N]``.  Should be placed at
            code edges (``c · LSB``) for floor semantics.
        drive_value: Legacy BL-clamp reference voltage in [V].  Used
            only by the deprecated ``drive(shape)`` shim.  New
            pipelines route the BL clamp through the OpAmpTIA.
        input_transform: ``"linear"`` (identity) or ``"log2"``.
        sampling_noise: Input-referred Gaussian sampling-stage noise.
        comparator_noise: Per-comparator threshold offset noise.
        drive_thermal: Gaussian thermal noise on the legacy drive
            output.
        energy_per_op__fJ: Dynamic energy per conversion (constant).
        latency_per_op__ns: Conversion latency.
        leakage_per_inst__uW: Static leakage power per instance.
        area_per_inst__um2: Silicon area per instance.
    """

    boundaries: list[float]

    drive_value: float = 0.0

    sampling_noise: float | None = None
    comparator_noise: float | None = None
    drive_thermal: float | None = None

    input_transform: Literal["linear", "log2"] = "linear"

    energy_per_op__fJ: float = 0.0
    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0


class GeneralADC(ADC):
    """Boundary-bucketize ADC with three Gaussian noise stages.

    Pipeline inside :meth:`convert`:

    1. Compute differential signal ``v_pos__V - v_neg__V``.
    2. Sampling noise (input-referred jitter).
    3. Optional ``log2`` domain transform.
    4. Comparator noise (per-threshold offset).
    5. Optional uniform jitter for stochastic rounding.
    6. ``floor_bucketize`` against ``boundaries`` → ``int16`` code.

    Registered non-persistent buffers: ``boundaries``, ``drive_value``.

    Single-mode contract: ``convert(..., mode, bits)`` validates
    ``mode == 0`` and ``bits == self._n_bits`` (the bit width implied
    by the boundary list); other values raise ``ValueError``.

    Args:
        config: Immutable ADC configuration.
        stochastic: Per-instance switch for stochastic rounding in
            :meth:`convert`.  ``None`` (default) follows
            ``module.training``; ``True`` / ``False`` force on / off.
        dtype: Float dtype for the boundary / drive buffers.
    """

    boundaries: Tensor
    drive_value: Tensor

    def __init__(
        self,
        config: GeneralADCConfig,
        *,
        name: str = "",
        stochastic: bool | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.stochastic: bool | None = stochastic

        boundaries_t = torch.tensor(config.boundaries, dtype=dtype)
        if boundaries_t.numel() < 1:
            raise ValueError("GeneralADCConfig.boundaries must contain at least one threshold")
        self.register_buffer("boundaries", boundaries_t, persistent=False)

        self.register_buffer("drive_value", torch.tensor(config.drive_value, dtype=dtype), persistent=False)

        n_codes = boundaries_t.numel() + 1
        self._n_bits: int = max(math.ceil(math.log2(n_codes)), 1)

        # Approximate uniform LSB used as the stochastic-jitter scale.
        # For non-uniform boundary lists the average spacing is the
        # best zero-knowledge estimate; calibrated lists are typically
        # close to uniform.
        if boundaries_t.numel() >= 2:
            self._lsb_estimate: float = float((boundaries_t[1:] - boundaries_t[:-1]).mean().item())
        else:
            self._lsb_estimate = float(boundaries_t.item())

    # --- ADC interface --- #

    @property
    def area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def latency_per_op__ns(self, *, bits: int) -> float:
        if bits != self._n_bits:
            raise ValueError(f"GeneralADC: bits ({bits}) must equal self._n_bits ({self._n_bits})")
        return self.config.latency_per_op__ns

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        mode: int,
        bits: int,
    ) -> Tensor:
        """Quantise a differential analog voltage into an integer code via floor-bucketize.

        Args:
            v_pos__V: Positive-side analog input voltage [V].  Any shape.
            v_neg__V: Negative-side analog input voltage [V].  Same shape.
            mode: Runtime operating-point index — must be ``0``.
            bits: Active bit width — must equal the boundary-implied
                bit width.

        Returns:
            ``int16`` code tensor shaped like ``v_pos__V``.  Dynamic
            energy / latency emitted through the profiler side channel.
        """
        self._validate_runtime_args(mode, bits)
        signal = v_pos__V - v_neg__V

        if self.config.sampling_noise is not None:
            signal = apply_gaussian(signal, self.config.sampling_noise)

        if self.config.input_transform == "log2":
            signal = torch.log2(signal.clamp_min(1e-12))

        if self.config.comparator_noise is not None:
            signal = apply_gaussian(signal, self.config.comparator_noise)

        code = floor_bucketize(
            signal,
            self.boundaries,
            out_dtype=torch.int16,
            training=self.training,
            override=self.stochastic,
            lsb=self._lsb_estimate,
        )

        dynamic_energy__fJ = torch.full_like(code, self.config.energy_per_op__fJ, dtype=torch.float32)
        self._log_dynamic(dynamic_energy__fJ, self.config.latency_per_op__ns)
        return code

    # --- legacy drive() shim (OpAmpTIA replaces this in new pipelines) --- #

    def drive(self, shape: tuple[int, ...]) -> Tensor:
        """Legacy BL-clamp reference voltage broadcast.

        Kept for backward compatibility with pre-OpAmpTIA xbars.  New
        pipelines should obtain the BL clamp from
        :class:`neurox.analog.opamp_tia.OpAmpTIA` and let the ADC be a pure
        converter.

        Args:
            shape: Output shape.

        Returns:
            ``drive_value`` tensor broadcast to ``shape`` with optional
            Gaussian thermal noise.
        """
        signal = self.drive_value.expand(shape)
        if self.config.drive_thermal is not None:
            signal = apply_gaussian(signal, self.config.drive_thermal)
        return signal

    # --- shared helpers --- #

    def _validate_runtime_args(self, mode: int, bits: int) -> None:
        """Single-mode runtime validator: only ``(0, self._n_bits)`` is legal."""
        if mode != 0:
            raise ValueError(f"GeneralADC: mode ({mode}) must be 0 (single-mode behavioural ADC)")
        if bits != self._n_bits:
            raise ValueError(
                f"GeneralADC: bits ({bits}) must equal self._n_bits ({self._n_bits}) "
                f"— boundary-bucketize is locked to its calibrated grid"
            )
