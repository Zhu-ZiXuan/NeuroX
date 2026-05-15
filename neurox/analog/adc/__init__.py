"""ADC family for NeuroX CiM macros.

The whole family is **runtime multi-mode**: ``convert`` and
``latency_per_op__ns`` take explicit ``mode`` and ``bits`` keyword
arguments at every call site.  Single-mode behavioural subclasses
(``GeneralADC`` + the flash-style variants) honour the contract by
validating ``mode == 0`` and ``bits == max_bits``; multi-mode SAR
variants accept any ``(mode, bits)`` in their configured envelope.

The ADC family is **digitisation-only**: mapping codes back to
ideal-integer scale is a macro concern (see
:meth:`neurox.xbar.base.Xbar.output_rescale_factor`), so no ADC
exposes a ``rescale_factor`` method.

Exports:

* :class:`ADC` — abstract base.  All concrete ADCs inherit and
  implement ``convert``, ``latency_per_op__ns``,
  ``area_per_inst__um2``, ``leakage_per_inst__uW``.
* :class:`ADCMode` — one ``(n_bits, n_states, max_signal)``
  operating mode for the **flash-style** multi-mode subclasses
  (Pipeline, Cyclic, Ramp, GeneralADC's helper).  SAR uses its own
  ``(max_bits, v_refs)`` config and does not consume
  :class:`ADCMode`.
* :class:`GeneralADC` / :class:`GeneralADCConfig` —
  boundary-bucketize ADC with three Gaussian noise stages; the
  calibration-friendly baseline.
* :class:`McsSarAdc` / :class:`McsSarAdcConfig` — V_cm-based /
  Merged Capacitor Switching differential SAR.  The default SAR
  variant for new work.
* :class:`SarAdcMono` / :class:`SarAdcMonoConfig` — monotonic
  (Set-and-Down) differential SAR.  Currently a placeholder; the
  differential convert kernel is future work.
* :class:`PipelineADC` / :class:`PipelineADCConfig` — N-stage
  residual-amp pipeline.
* :class:`CyclicADC` / :class:`CyclicADCConfig` — single-stage
  cyclic algorithmic ADC.
* :class:`RampADC` / :class:`RampADCConfig` — counter-ramp ADC.
"""

from .base import ADC, ADCMode
from .general import GeneralADC, GeneralADCConfig
from .mcs_sar import McsSarAdc, McsSarAdcConfig
from .sar_mono import SarAdcMono, SarAdcMonoConfig

__all__ = [
    "ADC",
    "ADCMode",
    "GeneralADC",
    "GeneralADCConfig",
    "McsSarAdc",
    "McsSarAdcConfig",
    "SarAdcMono",
    "SarAdcMonoConfig",
]
