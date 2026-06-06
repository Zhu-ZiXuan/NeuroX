"""Shared macro helpers for the example scripts.

The macro config TOML is polymorphically resolved via ``_neurox_type``;
the same factory works for the registered XbarMacro family members:

- ``DirectXbarMacroConfig`` / ``InterArraySliceXbarMacroConfig`` /
  ``IntraArraySliceXbarMacroConfig`` (xbar-using) with either a physical
  or an :class:`IdealXbarConfig` inside ``xbar_config``.
- ``IdealXbarMacroConfig`` (degenerate; no xbar — pure integer matmul baseline).
"""

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch

from neurox.analog import AnalogMuxPolicy, DriverPolicy, SwitchCapPolicy
from neurox.analog.adc import (
    ADCPolicy,
    GeneralADCConfig,
    GeneralADCPolicy,
    McsSarAdcConfig,
    McsSarAdcPolicy,
    SarAdcMonoConfig,
    SarAdcMonoPolicy,
)
from neurox.analog.dac import GeneralDACPolicy
from neurox.analog.tia import OpAmpTIAPolicy
from neurox.common import T_ROOM__K, dataclass_from_file
from neurox.device import NMOSPolicy, RRAMPolicy
from neurox.macro import NeuroxMacroQuantMatMul
from neurox.macro.xbar import (
    DirectXbarMacroConfig,
    DirectXbarMacroPolicy,
    IdealXbarMacroConfig,
    IdealXbarMacroPolicy,
    InterArraySliceXbarMacroConfig,
    InterArraySliceXbarMacroPolicy,
    IntraArraySliceXbarMacroConfig,
    IntraArraySliceXbarMacroPolicy,
    XbarMacro,
    XbarMacroConfig,
    XbarMacroPolicy,
)
from neurox.operator import QuantSpec
from neurox.xbar import (
    IdealXbarConfig,
    IdealXbarPolicy,
    Offset1T1RXbarConfig,
    Offset1T1RXbarPolicy,
    XbarPolicy,
)
from neurox.xbar._1t1r import CircuitCore1T1RPolicy
from neurox.xbar.readout import OffsetSwitchCapMuxAdcReadOutConfig, OffsetSwitchCapMuxAdcReadOutPolicy

# Use fp32 in the physical 1T1R path to keep solver-side arithmetic stable.
_CIRCUIT_DTYPE = torch.float32

# Operator construction passes its own weight shape into the factory; the
# probe path needs a non-empty placeholder just to read value ranges.
_PROBE_W_LOGICAL_SHAPE = (1, 1)


@cache
def read_macro_config(config_path: Path) -> XbarMacroConfig:
    """Cached ``[macro]`` section, polymorphically resolved via ``_neurox_type``."""
    return dataclass_from_file(XbarMacroConfig, config_path, section="macro")


def supports_xbar_override(config: XbarMacroConfig) -> bool:
    """Whether ``--xbar physical|ideal`` is meaningful for ``config``.

    True only when the config carries a physical xbar (i.e. is xbar-using
    and ``config.xbar_config`` is not already an :class:`IdealXbarConfig`).
    """
    if isinstance(config, IdealXbarMacroConfig):
        return False
    xbar_config = getattr(config, "xbar_config", None)
    if xbar_config is None:
        return False
    return not isinstance(xbar_config, IdealXbarConfig)


def derive_quant_spec(config_path: Path) -> QuantSpec:
    """Derive the operator quantization grid from the macro ranges."""
    macro = _build_macro(
        config_path,
        name="probe",
        w_logical_shape=_PROBE_W_LOGICAL_SHAPE,
        ideal_xbar=False,
    )
    x_qmin, x_qmax = macro.x_value_range
    w_qmin, w_qmax = macro.w_value_range
    assert w_qmin == -w_qmax, f"Expected symmetric w_value_range, got ({w_qmin}, {w_qmax})"
    return QuantSpec(x_qmin=x_qmin, x_qmax=x_qmax, w_qmax=w_qmax, y_qmin=-x_qmax, y_qmax=x_qmax)


def _all_off_adc_policy(config: object) -> ADCPolicy:
    """Construct an all-False policy matching the concrete ADC config type."""
    if isinstance(config, GeneralADCConfig):
        return GeneralADCPolicy(sampling_noise=False, comparator_noise=False, drive_thermal=False)
    if isinstance(config, SarAdcMonoConfig):
        return SarAdcMonoPolicy(
            cap_mismatch=False, comparator_offset=False, comparator_thermal_noise=False, sampling_thermal_noise=False
        )
    if isinstance(config, McsSarAdcConfig):
        return McsSarAdcPolicy(
            cap_mismatch=False, comparator_offset=False, comparator_thermal_noise=False, sampling_thermal_noise=False
        )
    raise TypeError(f"no all-off policy registered for ADC config type {type(config).__name__}")


def _all_off_xbar_policy(config: object) -> XbarPolicy:
    """Construct an all-False policy matching the concrete Xbar config type."""
    if isinstance(config, IdealXbarConfig):
        return IdealXbarPolicy()
    if isinstance(config, Offset1T1RXbarConfig):
        readout_config = config.readout_config
        if not isinstance(readout_config, OffsetSwitchCapMuxAdcReadOutConfig):
            raise TypeError(f"no all-off readout policy for {type(readout_config).__name__}")
        return Offset1T1RXbarPolicy(
            core=CircuitCore1T1RPolicy(
                rram=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
                nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
                tia=OpAmpTIAPolicy(
                    opamp_gain_sigma=False,
                    nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
                ),
                sl_driver=DriverPolicy(drive_thermal=False),
                wl_dac=GeneralDACPolicy(drive_thermal=False),
            ),
            readout=OffsetSwitchCapMuxAdcReadOutPolicy(
                data_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
                ref_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
                analog_mux=AnalogMuxPolicy(mux_noise_cm=False, mux_noise_dm=False),
                bl_adc=_all_off_adc_policy(readout_config.adc_config),
            ),
        )
    raise TypeError(f"no all-off xbar policy for {type(config).__name__}")


def _all_off_macro_policy(config: XbarMacroConfig) -> XbarMacroPolicy:
    """Construct an all-False policy matching the concrete XbarMacro config type."""
    if isinstance(config, IdealXbarMacroConfig):
        return IdealXbarMacroPolicy()
    if isinstance(config, DirectXbarMacroConfig):
        return DirectXbarMacroPolicy(xbar=_all_off_xbar_policy(config.xbar_config))
    if isinstance(config, InterArraySliceXbarMacroConfig):
        return InterArraySliceXbarMacroPolicy(xbar=_all_off_xbar_policy(config.xbar_config))
    if isinstance(config, IntraArraySliceXbarMacroConfig):
        return IntraArraySliceXbarMacroPolicy(xbar=_all_off_xbar_policy(config.xbar_config))
    raise TypeError(f"no all-off macro policy for {type(config).__name__}")


def _build_macro(
    config_path: Path,
    *,
    name: str,
    w_logical_shape: tuple[int, ...],
    ideal_xbar: bool,
) -> NeuroxMacroQuantMatMul:
    """Build one macro from a TOML config, dispatching on the config type.

    The macro is constructed with all nonidealities disabled. Callers that
    want to study noise must build their own policy and call
    ``XbarMacro.from_config`` directly.

    ``ideal_xbar`` is honoured only by xbar-using members (swaps the
    physical xbar for its ideal twin); :class:`IdealXbarMacro` ignores it.
    """
    config = read_macro_config(config_path)
    return XbarMacro.from_config(
        config=config,
        policy=_all_off_macro_policy(config),
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=_CIRCUIT_DTYPE,
        T__K=T_ROOM__K,
        ideal_xbar=ideal_xbar,
    )


def build_macro_factory(config_path: Path, *, ideal_xbar: bool) -> Callable[..., NeuroxMacroQuantMatMul]:
    """Return the per-layer macro builder for the requested TOML + xbar flavour.

    Args:
        config_path: TOML path resolved against ``example/<model>/`` by the caller.
        ideal_xbar: Forwarded to the macro; relevant only when the config
            carries a physical xbar (see :func:`supports_xbar_override`).
    """

    def factory(*, name: str, w_logical_shape: tuple[int, ...]) -> NeuroxMacroQuantMatMul:
        return _build_macro(
            config_path,
            name=name,
            w_logical_shape=w_logical_shape,
            ideal_xbar=ideal_xbar,
        )

    return factory


__all__ = [
    "build_macro_factory",
    "derive_quant_spec",
    "read_macro_config",
    "supports_xbar_override",
]
