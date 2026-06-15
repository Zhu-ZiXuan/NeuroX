"""Local macro factory for the BERT example. Self-contained, no shared code.

Boilerplate that walks the chip preset's config-type tree and builds the
corresponding all-False nonideality policy, then instantiates the macro.
Used by ``BERT model_quant.py`` to construct one macro per layer.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path

import torch

from neurox.analog import AnalogMuxPolicy, DriverPolicy, SwitchCapPolicy
from neurox.analog.adc import (
    ADCConfig,
    ADCPolicy,
    GeneralADCConfig,
    GeneralADCPolicy,
    McsSarAdcConfig,
    McsSarAdcPolicy,
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
from neurox.xbar import (
    IdealXbarConfig,
    IdealXbarPolicy,
    Offset1T1RXbarConfig,
    Offset1T1RXbarPolicy,
    XbarConfig,
    XbarPolicy,
)
from neurox.xbar._1t1r import CircuitCore1T1RPolicy
from neurox.xbar.readout import OffsetSwitchCapMuxAdcReadOutConfig, OffsetSwitchCapMuxAdcReadOutPolicy

_CIRCUIT_DTYPE = torch.float32


@cache
def read_macro_config(config_path: Path) -> XbarMacroConfig:
    return dataclass_from_file(XbarMacroConfig, config_path, section="macro")


def _all_off_adc_policy(config: ADCConfig) -> ADCPolicy:
    if isinstance(config, GeneralADCConfig):
        return GeneralADCPolicy(sampling_noise=False, comparator_noise=False, drive_thermal=False)
    if isinstance(config, McsSarAdcConfig):
        return McsSarAdcPolicy(
            cap_mismatch=False, comparator_offset=False, comparator_thermal_noise=False, sampling_thermal_noise=False
        )
    raise TypeError(f"no all-off policy registered for ADC config type {type(config).__name__}")


def _all_off_xbar_policy(config: XbarConfig, *, solve_chunk_size_x: int, solve_chunk_size_inst: int) -> XbarPolicy:
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
                solve_chunk_size_x=solve_chunk_size_x,
                solve_chunk_size_inst=solve_chunk_size_inst,
            ),
            readout=OffsetSwitchCapMuxAdcReadOutPolicy(
                data_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
                ref_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
                analog_mux=AnalogMuxPolicy(mux_noise_cm=False, mux_noise_dm=False),
                bl_adc=_all_off_adc_policy(readout_config.adc_config),
            ),
        )
    raise TypeError(f"no all-off xbar policy for {type(config).__name__}")


def _all_off_macro_policy(
    config: XbarMacroConfig,
    *,
    solve_chunk_size_x: int,
    solve_chunk_size_inst: int,
) -> XbarMacroPolicy:
    if isinstance(config, IdealXbarMacroConfig):
        return IdealXbarMacroPolicy()
    if not isinstance(config, (DirectXbarMacroConfig, InterArraySliceXbarMacroConfig, IntraArraySliceXbarMacroConfig)):
        raise TypeError(f"no all-off macro policy for {type(config).__name__}")
    xbar_policy = _all_off_xbar_policy(
        config.xbar_config,
        solve_chunk_size_x=solve_chunk_size_x,
        solve_chunk_size_inst=solve_chunk_size_inst,
    )
    if isinstance(config, DirectXbarMacroConfig):
        return DirectXbarMacroPolicy(xbar=xbar_policy)
    if isinstance(config, InterArraySliceXbarMacroConfig):
        return InterArraySliceXbarMacroPolicy(xbar=xbar_policy)
    return IntraArraySliceXbarMacroPolicy(xbar=xbar_policy)


def build_macro_factory(
    config_path: Path,
    *,
    ideal_xbar: bool,
    solve_chunk_size_x: int = 0,
    solve_chunk_size_inst: int = 0,
) -> Callable[..., NeuroxMacroQuantMatMul]:
    """Return ``(name, w_logical_shape) → macro`` for the given TOML.

    ``ideal_xbar=True`` swaps the physical xbar for its ideal twin
    (only meaningful when the TOML carries a physical xbar config).
    """

    def factory(*, name: str, w_logical_shape: tuple[int, ...]) -> NeuroxMacroQuantMatMul:
        config = read_macro_config(config_path)
        return XbarMacro.from_config(
            config=config,
            policy=_all_off_macro_policy(
                config,
                solve_chunk_size_x=solve_chunk_size_x,
                solve_chunk_size_inst=solve_chunk_size_inst,
            ),
            name=name,
            w_logical_shape=w_logical_shape,
            dtype=_CIRCUIT_DTYPE,
            T__K=T_ROOM__K,
            ideal_xbar=ideal_xbar,
        )

    return factory
