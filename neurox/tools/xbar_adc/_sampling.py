"""Shared private backend for ``neurox/tools/xbar_adc/`` CLIs.

See also:
    docs/dev/modules/tools/xbar_adc/README.md
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.analog import AnalogMuxPolicy, DriverPolicy, SwitchCapPolicy
from neurox.analog.adc import (
    ADCConfig,
    ADCPolicy,
    GeneralADCConfig,
    GeneralADCPolicy,
    McsSarAdcConfig,
    McsSarAdcPolicy,
)
from neurox.analog.dac import DACConfig, DACPolicy, GeneralDACConfig, GeneralDACPolicy
from neurox.analog.tia import OpAmpTIAConfig, OpAmpTIAPolicy, TIAConfig, TIAPolicy
from neurox.common import T_ROOM__K, dataclass_from_file, dict_from_file
from neurox.device import NMOSPolicy, RRAMPolicy
from neurox.xbar import Offset1T1RXbar, Offset1T1RXbarConfig, Offset1T1RXbarPolicy
from neurox.xbar._1t1r import CircuitCore1T1RPolicy
from neurox.xbar.readout import OffsetSwitchCapMuxAdcReadOutConfig, OffsetSwitchCapMuxAdcReadOutPolicy


@dataclass(frozen=True)
class Distribution:
    """Parsed synthetic-workload distribution.

    ``None``-valued fields mean uniform sampling over the xbar's
    legal integer range on that axis.

    Attributes:
        w_values: Allowed per-cell digit values, int64 1-D, or ``None``.
        w_probs: Normalized probabilities matching ``w_values``,
            float64 1-D, or ``None``.
        x_values: Allowed per-row input codes, int64 1-D, or ``None``.
        x_probs: Normalized probabilities matching ``x_values``,
            float64 1-D, or ``None``.
        source: Human-readable provenance — ``"uniform"`` or the TOML path.
    """

    w_values: Tensor | None
    w_probs: Tensor | None
    x_values: Tensor | None
    x_probs: Tensor | None
    source: str


def load_distribution(path: Path | None, xbar: Offset1T1RXbar) -> Distribution:
    """Load a synthetic-workload distribution TOML.

    ``None`` means fully uniform; a present file may omit ``[w]`` or
    ``[x]`` to keep that axis uniform.

    Schema::

        [w]
        values = [-1, 0, 1]
        probs = [0.20, 0.65, 0.15]

        [x]
        values = [0, 1, 2, 3]
        probs = [0.70, 0.20, 0.08, 0.02]

    Args:
        path: TOML path or ``None``.
        xbar: Built xbar — supplies ``w_digit_range`` / ``x_range`` for
            value-set validation.

    Returns:
        Validated, internally-normalised :class:`Distribution`.

    Raises:
        ValueError: When a section is malformed (length mismatch,
            negative probability, all-zero probability, value outside
            the xbar's legal range, etc.).
    """
    if path is None:
        return Distribution(
            w_values=None,
            w_probs=None,
            x_values=None,
            x_probs=None,
            source="uniform",
        )

    raw = dict_from_file(path)
    unknown = sorted(k for k in raw if k not in ("w", "x"))
    if unknown:
        raise ValueError(
            f"distribution TOML at {path}: unknown top-level key(s) {unknown}; only '[w]' and '[x]' are recognised"
        )
    w_values, w_probs = _load_axis(raw, "w", xbar.w_digit_range)
    x_values, x_probs = _load_axis(raw, "x", xbar.x_range)
    return Distribution(
        w_values=w_values,
        w_probs=w_probs,
        x_values=x_values,
        x_probs=x_probs,
        source=str(path),
    )


def _load_axis(
    raw: dict[str, Any],
    key: str,
    legal_range: tuple[int, int],
) -> tuple[Tensor | None, Tensor | None]:
    """Return ``(values_int64, normalized_probs_float64)`` or ``(None, None)``."""
    if key not in raw:
        return None, None

    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(f"distribution [{key}]: expected a TOML table, got {type(section).__name__}")
    if "values" not in section or "probs" not in section:
        raise ValueError(f"distribution [{key}]: both 'values' and 'probs' are required")

    values_raw = section["values"]
    probs_raw = section["probs"]

    if not isinstance(values_raw, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in values_raw):
        raise ValueError(f"distribution [{key}].values: must be a list of integers")
    if not isinstance(probs_raw, list) or not all(
        isinstance(p, (int, float)) and not isinstance(p, bool) for p in probs_raw
    ):
        raise ValueError(f"distribution [{key}].probs: must be a list of numbers")

    if len(values_raw) == 0:
        raise ValueError(f"distribution [{key}].values: must be non-empty")
    if len(values_raw) != len(probs_raw):
        raise ValueError(f"distribution [{key}]: len(values)={len(values_raw)} != len(probs)={len(probs_raw)}")

    probs = [float(p) for p in probs_raw]
    if any(p < 0.0 for p in probs):
        raise ValueError(f"distribution [{key}].probs: every entry must be >= 0")
    total = sum(probs)
    if total <= 0.0:
        raise ValueError(f"distribution [{key}].probs: at least one entry must be > 0")

    lo, hi = legal_range
    for v in values_raw:
        if not (lo <= v <= hi):
            raise ValueError(f"distribution [{key}].values: value {v} outside xbar legal range [{lo}, {hi}]")

    values = torch.tensor(values_raw, dtype=torch.int64)
    probs_tensor = torch.tensor([p / total for p in probs], dtype=torch.float64)
    return values, probs_tensor


def sample_w(
    distribution: Distribution,
    xbar: Offset1T1RXbar,
    *,
    n: int,
    batch_w: int = 1,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield ``n // batch_w`` batches of independent xbar-native digit tensors.

    ``n`` **must** be a multiple of ``batch_w`` — every yielded tensor
    carries a fixed leading ``(batch_w,)`` axis to match the xbar's
    ``inst_shape=(batch_w,)`` contract, so a partial final batch would
    immediately fail ``xbar.program(w)``'s shape check. Round ``n`` up
    to the next multiple of ``batch_w`` at the call site if you need
    "at least N" coverage.

    With ``batch_w == 1`` (default), each yielded tensor matches
    :attr:`_w_layout_shape` for an ``inst_shape=()`` xbar, i.e.
    ``(col_num, w_digit_count, row_num)``. With ``batch_w > 1``, each
    yield is ``(batch_w, col_num, w_digit_count, row_num)``.
    """
    if batch_w <= 0:
        raise ValueError(f"batch_w ({batch_w}) must be > 0")
    if n % batch_w != 0:
        raise ValueError(
            f"sample_w: n ({n}) must be a multiple of batch_w ({batch_w}); "
            "partial final batches would break the xbar's fixed inst_shape contract"
        )
    leading = () if batch_w == 1 else (batch_w,)
    shape_per = (*leading, xbar.col_num, xbar.w_digit_count, xbar.row_num)
    n_per = math.prod(shape_per)
    num_yields = n // batch_w
    for _ in range(num_yields):
        if distribution.w_values is None:
            lo, hi = xbar.w_digit_range
            yield torch.randint(
                lo,
                hi + 1,
                shape_per,
                device=device,
                dtype=torch.int64,
                generator=generator,
            )
        else:
            assert distribution.w_probs is not None  # paired with w_values via _load_axis
            probs = distribution.w_probs.to(device)
            values = distribution.w_values.to(device)
            idx = torch.multinomial(probs, num_samples=n_per, replacement=True, generator=generator)
            yield values[idx].view(shape_per).to(torch.int64)


def sample_x_batches(
    distribution: Distribution,
    xbar: Offset1T1RXbar,
    *,
    n_total: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield input batches summing to ``n_total`` vectors.

    Each batch has shape ``(min(batch_size, remaining), row_num)``,
    int64, on ``device``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be > 0")
    if n_total < 0:
        raise ValueError(f"n_total ({n_total}) must be >= 0")
    row_num = xbar.row_num
    remaining = n_total
    while remaining > 0:
        cur = min(batch_size, remaining)
        if distribution.x_values is None:
            lo, hi = xbar.x_range
            x = torch.randint(
                lo,
                hi + 1,
                (cur, row_num),
                device=device,
                dtype=torch.int64,
                generator=generator,
            )
        else:
            assert distribution.x_probs is not None  # paired with x_values via _load_axis
            probs = distribution.x_probs.to(device)
            values = distribution.x_values.to(device)
            idx = torch.multinomial(
                probs,
                num_samples=cur * row_num,
                replacement=True,
                generator=generator,
            )
            x = values[idx].view(cur, row_num).to(torch.int64)
        yield x
        remaining -= cur


def build_offset_1t1r_xbar_all_off_from_config(
    xbar_config: Offset1T1RXbarConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    inst_shape: tuple[int, ...] = (),
    solve_chunk_size_x: int = 0,
    solve_chunk_size_inst: int = 0,
) -> Offset1T1RXbar:
    """Build a fully nonideality-free :class:`Offset1T1RXbar` from a dataclass.

    Same semantics as :func:`build_offset_1t1r_xbar_all_off` but takes an
    already-loaded :class:`Offset1T1RXbarConfig`; use this when the xbar
    config is nested inside a larger tool-run TOML pulled in via
    ``_neurox_use``.

    Raises:
        TypeError: When the readout or ADC config is not currently
            supported by the ``xbar_adc`` tool family.
    """
    readout_config = xbar_config.readout_config
    if not isinstance(readout_config, OffsetSwitchCapMuxAdcReadOutConfig):
        raise TypeError(
            f"unsupported readout config type: {type(readout_config).__name__}; "
            "tools in neurox.tools.xbar_adc only support OffsetSwitchCapMuxAdcReadOut."
        )

    bl_adc_policy = _all_off_adc_policy(readout_config.adc_config)
    tia_policy = _all_off_tia_policy(xbar_config.core_config.tia_config)
    wl_dac_policy = _all_off_dac_policy(xbar_config.core_config.wl_dac_config)
    policy = Offset1T1RXbarPolicy(
        core=CircuitCore1T1RPolicy(
            rram=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
            nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
            tia=tia_policy,
            sl_driver=DriverPolicy(drive_thermal=False),
            wl_dac=wl_dac_policy,
            solve_chunk_size_x=solve_chunk_size_x,
            solve_chunk_size_inst=solve_chunk_size_inst,
        ),
        readout=OffsetSwitchCapMuxAdcReadOutPolicy(
            data_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
            ref_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
            analog_mux=AnalogMuxPolicy(mux_noise_cm=False, mux_noise_dm=False),
            bl_adc=bl_adc_policy,
        ),
    )

    xbar = Offset1T1RXbar(
        config=xbar_config,
        policy=policy,
        name="xbar",
        inst_shape=inst_shape,
        dtype=dtype,
        T__K=T_ROOM__K,
    )
    xbar.to(device)
    xbar.eval()
    xbar.fabricate()
    return xbar


def build_offset_1t1r_xbar_all_off(
    config_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    inst_shape: tuple[int, ...] = (),
    solve_chunk_size_x: int = 0,
    solve_chunk_size_inst: int = 0,
) -> Offset1T1RXbar:
    """Build a fully nonideality-free :class:`Offset1T1RXbar` from a TOML.

    Every nonideality flag in the policy tree is set to ``False``, giving
    a deterministic nominal physics path appropriate for ADC range
    exploration and ``rescale_factor`` calibration.

    Args:
        config_path: Path to a chip TOML carrying a ``[xbar]`` section
            (``_neurox_type = "Offset1T1RXbarConfig"``).
        device: Target torch device for buffer placement.
        dtype: Internal float dtype.
        solve_chunk_size_x: ``CircuitCore1T1RPolicy.solve_chunk_size_x``; ``0`` disables.
        solve_chunk_size_inst: ``CircuitCore1T1RPolicy.solve_chunk_size_inst``; ``0`` disables.
    """
    xbar_config = dataclass_from_file(Offset1T1RXbarConfig, config_path, section="xbar")
    return build_offset_1t1r_xbar_all_off_from_config(
        xbar_config,
        device=device,
        dtype=dtype,
        inst_shape=inst_shape,
        solve_chunk_size_x=solve_chunk_size_x,
        solve_chunk_size_inst=solve_chunk_size_inst,
    )


def _all_off_adc_policy(adc_config: ADCConfig) -> ADCPolicy:
    """Build the matching ADCPolicy for ``adc_config`` with every flag ``False``.

    ``SarAdcMonoConfig`` is intentionally rejected: :class:`SarAdcMono.convert`
    is not implemented yet, so accepting it here would let the tool succeed
    at setup and fail mid-sweep with ``NotImplementedError``.
    """
    if isinstance(adc_config, GeneralADCConfig):
        return GeneralADCPolicy(sampling_noise=False, comparator_noise=False, drive_thermal=False)
    if isinstance(adc_config, McsSarAdcConfig):
        return McsSarAdcPolicy(
            cap_mismatch=False,
            comparator_offset=False,
            comparator_thermal_noise=False,
            sampling_thermal_noise=False,
        )
    raise TypeError(
        f"unsupported adc config type: {type(adc_config).__name__}; "
        "neurox.tools.xbar_adc supports GeneralADC and McsSarAdc only"
    )


def _all_off_tia_policy(tia_config: TIAConfig) -> TIAPolicy:
    """Build the matching TIAPolicy for ``tia_config`` with every flag ``False``."""
    if isinstance(tia_config, OpAmpTIAConfig):
        return OpAmpTIAPolicy(
            opamp_gain_sigma=False,
            nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        )
    raise TypeError(
        f"unsupported tia config type: {type(tia_config).__name__}; neurox.tools.xbar_adc supports OpAmpTIA only"
    )


def _all_off_dac_policy(dac_config: DACConfig) -> DACPolicy:
    """Build the matching DACPolicy for ``dac_config`` with every flag ``False``."""
    if isinstance(dac_config, GeneralDACConfig):
        return GeneralDACPolicy(drive_thermal=False)
    raise TypeError(
        f"unsupported dac config type: {type(dac_config).__name__}; neurox.tools.xbar_adc supports GeneralDAC only"
    )


def make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    """Build a deterministic generator on ``device`` when ``seed`` is set."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
