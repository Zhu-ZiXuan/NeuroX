"""Derive the Xue2020 ADC ladder and calibrate read-path energy terms.

The fixed cell and CABLC models are inputs to this campaign. It derives the ADC
reference ladder, then fits PN decision energy and the TMCSA conduction scale at
the declared workload. Results are written only to the requested report.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import sys
import tomllib
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from neurox import stamp_names
from neurox.primitive.analog.current_adc import IadcProber
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rLinearConfig
from neurox.works.macro.cim.xue2020jssc import (
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)

_LOG = logging.getLogger(__name__)
_VAL_DIR = Path(__file__).resolve().parents[1]
_QUANTIZATION_MODE = 0
_CABLC_CHANNEL = ".cablc"
_ARRAY_ROW = "array"
_ROW_KEYS = (".control", _CABLC_CHANNEL, _ARRAY_ROW, "dswct", "sinwp_sc", "pn_isub", "tmcsa")
sys.path.insert(0, str(_VAL_DIR))
V = importlib.import_module("validate")


def rebuild(
    config: CimMacroConfig,
    policy: CimMacroPolicy,
    device: torch.device,
) -> Xue2020JsscCimMacro:
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=V.ROW_NUM,
        output_num=V.COL_NUM,
        inst_shape=(),
        dtype=torch.float32,
        T__K=V.TEMPERATURE__K,
    )
    if not isinstance(macro, Xue2020JsscCimMacro):
        raise TypeError(f"expected Xue2020JsscCimMacro, got {type(macro).__name__}")
    macro.to(device)
    macro.eval()
    macro.fabricate()
    stamp_names(macro)
    return macro


def set_conduction_scale(
    config: Xue2020JsscCimMacroConfig,
    scale: float,
) -> Xue2020JsscCimMacroConfig:
    return dataclasses.replace(
        config,
        tmcsa_config=dataclasses.replace(config.tmcsa_config, conduction_scale=scale),
    )


def set_pn_energy(
    config: Xue2020JsscCimMacroConfig,
    energy_per_op__fJ: float,
) -> Xue2020JsscCimMacroConfig:
    return dataclasses.replace(
        config,
        pn_isub_config=dataclasses.replace(config.pn_isub_config, e_per_op__fJ=energy_per_op__fJ),
    )


def _staircase_drive(macro: Xue2020JsscCimMacro) -> torch.Tensor:
    config = macro.config
    device = next(macro.buffers()).device
    x_max = (1 << config.input_bit_num) - 1
    m_max = (1 << config.adc_config.bits) - 1
    x = torch.zeros((m_max + 1, macro.row_num), dtype=torch.long, device=device)
    for magnitude in range(m_max + 1):
        remaining = magnitude
        for row in range(config.max_active_num):
            value = min(x_max, remaining)
            x[magnitude, row] = value
            remaining -= value
        if remaining:
            raise ValueError(f"MAC magnitude {magnitude} exceeds the drivable range ({config.max_active_num * x_max})")
    return x


def _program_unit_weight(macro: Xue2020JsscCimMacro) -> None:
    device = next(macro.buffers()).device
    weight = torch.zeros((macro.row_num, macro.col_num), dtype=torch.long, device=device)
    weight[:, 0] = 1
    macro.program(weight)


def unit_isub_staircase(macro: Xue2020JsscCimMacro) -> list[float]:
    _program_unit_weight(macro)
    x = _staircase_drive(macro)
    with IadcProber(sync_device=torch.device("cpu")) as prober, torch.no_grad():
        macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=macro.adc_max_bits)
    i_sub__uA = prober.records[-1].i_in__uA
    while i_sub__uA.ndim > 3:
        i_sub__uA = i_sub__uA[0]
    return [float(value) for value in i_sub__uA[:, 0, 0].cpu()]


def midpoint_ladder(grid: list[float]) -> list[float]:
    return [0.5 * (lower + upper) for lower, upper in pairwise(grid)]


def round_significant(value: float, digits: int = 5) -> float:
    return float(f"{value:.{digits}g}")


def verify_ladder(macro: Xue2020JsscCimMacro) -> tuple[bool, list[int]]:
    _program_unit_weight(macro)
    x = _staircase_drive(macro)
    with torch.no_grad():
        codes = macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=macro.adc_max_bits)
    actual = [int(codes[magnitude, 0]) for magnitude in range(x.shape[0])]
    return actual == list(range(x.shape[0])), actual


def measure_rows(
    macro: Xue2020JsscCimMacro,
    anchors: dict[str, Any],
    *,
    n_w: int,
    n_x: int,
    repeat: int,
    seed: int,
) -> dict[str, float]:
    measurement = V.measure_rounds(
        macro,
        anchors,
        n_w=n_w,
        n_x=n_x,
        repeat=repeat,
        seed=seed,
    )
    return {key: measurement.dyn_by_name.get(key, 0.0) for key in _ROW_KEYS}


def pn_energy_per_op(
    rows: dict[str, float],
    *,
    pair_target__fJ: float,
    current_energy_per_op__fJ: float,
    io_num: int,
) -> float:
    measured__fJ = rows["sinwp_sc"] + rows["pn_isub"]
    current_fixed__fJ = current_energy_per_op__fJ * io_num
    conduction__fJ = measured__fJ - current_fixed__fJ
    fitted__fJ = (pair_target__fJ - conduction__fJ) / io_num
    if fitted__fJ < 0.0:
        raise ValueError("the SINWP-SC + PN-ISUB target leaves no non-negative PN decision energy")
    return fitted__fJ


def fitted_conduction_scale(
    rows: dict[str, float],
    *,
    target__fJ: float,
    fixed__fJ: float,
    current_scale: float,
) -> float:
    measured_conduction__fJ = rows["tmcsa"] - fixed__fJ
    target_conduction__fJ = target__fJ - fixed__fJ
    if measured_conduction__fJ <= 0.0:
        raise ValueError("TMCSA measured conduction energy must be positive before fitting its scale")
    if target_conduction__fJ <= 0.0:
        raise ValueError("the TMCSA target leaves no positive conduction-energy residual")
    return current_scale * target_conduction__fJ / measured_conduction__fJ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=_VAL_DIR / "params.toml")
    parser.add_argument("--policy", type=Path, default=_VAL_DIR / "policy.toml")
    parser.add_argument("--anchors", type=Path, default=_VAL_DIR / "anchors.toml")
    parser.add_argument("--n-w", type=int, default=4)
    parser.add_argument("--n-x", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--solve-chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--report", type=Path, default=_VAL_DIR / "calibration.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = V.resolve_device(args.device)
    with args.anchors.open("rb") as file:
        anchors = tomllib.load(file)

    config = CimMacroConfig.from_file(args.params, section="cim_macro")
    if not isinstance(config, Xue2020JsscCimMacroConfig):
        raise TypeError(f"expected Xue2020JsscCimMacroConfig, got {type(config).__name__}")
    policy = CimMacroPolicy.from_file(args.policy, section="policy")
    if not isinstance(policy, Xue2020JsscCimMacroPolicy):
        raise TypeError(f"expected Xue2020JsscCimMacroPolicy, got {type(policy).__name__}")
    policy = dataclasses.replace(
        policy,
        array_policy=dataclasses.replace(policy.array_policy, solve_chunk_size=args.solve_chunk),
    )

    target__fJ = float(anchors["target"]["per_access__fJ"])
    shares = anchors["fig18_shares"]
    pair1_target__fJ = float(shares["cablc"] + shares["dswct"]) / 100.0 * target__fJ
    pair2_target__fJ = float(shares["sinwp_sc"] + shares["pn_isub"]) / 100.0 * target__fJ
    tmcsa_target__fJ = float(shares["tmcsa"]) / 100.0 * target__fJ

    report: list[str] = []

    def emit(line: str = "") -> None:
        _LOG.info("%s", line)
        report.append(line)

    emit("# xue2020jssc calibration campaign")
    emit()
    emit(
        f"Device {device}; n_w={args.n_w}, n_x={args.n_x}, repeat={args.repeat}, "
        f"seed={args.seed}; P(x!=0)={anchors['data']['input_nonzero_probability']:.4f}, "
        f"P(w!=0)={anchors['data']['weight_nonzero_probability']:.4f}. The script reports values only."
    )

    emit()
    emit("## 0. Activity calibration input")
    emit()
    emit(
        "The declared activity point comes from a prior joint sweep against the two Fig.18 read-path pair "
        "targets. The minimum-error point reaches the dense-weight boundary P(w!=0)=1.0 and uses "
        "P(x!=0)=0.2935 among at most nine candidate rows (about 2.64 nonzero values per input on average). "
        "Conditional nonzero magnitudes remain uniform. This script treats that effective activity point as "
        "an input; it does not claim that the paper reports either probability."
    )

    emit()
    emit("## 1. Fixed 1T1R cell input")
    emit()
    cell_config = config.array_config.cell_config
    if not isinstance(cell_config, XbarCell1t1rLinearConfig):
        raise TypeError(f"expected XbarCell1t1rLinearConfig, got {type(cell_config).__name__}")
    emit(f"WL-off g [uS]: {cell_config.g_cell_off_table__uS}.")
    emit(f"WL-on g [uS]: {cell_config.g_cell_on_table__uS}.")
    emit(f"WL-off V_X ratios: {cell_config.vx_ratio_off_table}.")
    emit(f"WL-on V_X ratios: {cell_config.vx_ratio_on_table}.")

    emit()
    emit(
        "Solver input: n_outer=4, n_inner=3. The separate residual campaign found a 3/3 plateau and added "
        "one outer margin."
    )

    emit()
    emit("## 2. ADC reference ladder")
    emit()
    macro = rebuild(config, policy, device)
    grid = unit_isub_staircase(macro)
    if not all(lower < upper for lower, upper in pairwise(grid)):
        raise ValueError("the probed I_SUB staircase is not strictly increasing")
    ladder = [round_significant(value) for value in midpoint_ladder(grid)]
    config = dataclasses.replace(
        config,
        reference_config=dataclasses.replace(config.reference_config, i_refs__uA=(tuple(ladder),)),
    )
    macro = rebuild(config, policy, device)
    ladder_ok, codes = verify_ladder(macro)
    emit(f"I_SUB [uA]: {', '.join(f'{value:.6f}' for value in grid)}")
    emit(f"I_REF [uA]: {', '.join(f'{value:.6f}' for value in ladder)}")
    emit(f"Decode check: {'PASS' if ladder_ok else 'FAIL'} ({codes}).")

    emit()
    emit("## 3. CABLC+DSWCT conduction")
    emit()
    rows = measure_rows(
        macro,
        anchors,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
        seed=args.seed,
    )
    input_conduction__fJ = rows[_CABLC_CHANNEL] + rows["dswct"]
    input_cap__fJ = rows[_ARRAY_ROW]
    emit(
        "Measured rows before fitting [fJ/access]: " + ", ".join(f"{name}={rows[name]:.4f}" for name in _ROW_KEYS) + "."
    )
    emit(
        f"Input conduction {input_conduction__fJ:.4f} fJ/access + "
        f"declared array capacitance {input_cap__fJ:.4f} fJ/access; "
        f"Fig.18 pair target {pair1_target__fJ:.4f} fJ/access."
    )
    emit("Array node caps remain the declared zero paper-reproduction seat; no residual capacitance is fitted.")

    emit()
    emit("## 4. PN-ISUB decision energy")
    emit()
    io_num = macro.col_num // config.mux_factor
    pair2_before__fJ = rows["sinwp_sc"] + rows["pn_isub"]
    pn_energy_raw__fJ = pn_energy_per_op(
        rows,
        pair_target__fJ=pair2_target__fJ,
        current_energy_per_op__fJ=config.pn_isub_config.e_per_op__fJ,
        io_num=io_num,
    )
    pn_energy__fJ = round_significant(pn_energy_raw__fJ)
    config = set_pn_energy(config, pn_energy__fJ)
    emit(
        f"Before fitting: SINWP-SC + PN-ISUB {pair2_before__fJ:.4f} fJ/access; "
        f"Fig.18 pair target {pair2_target__fJ:.4f} fJ/access."
    )
    emit(
        f"PN-ISUB residual {pn_energy_raw__fJ:.8g} fJ per (slot, IO); rounded decision energy {pn_energy__fJ:#.5g} fJ."
    )

    emit()
    emit("## 5. TMCSA conduction")
    emit()
    macro = rebuild(config, policy, device)
    rows = measure_rows(
        macro,
        anchors,
        n_w=args.n_w,
        n_x=args.n_x,
        repeat=args.repeat,
        seed=args.seed,
    )
    fitted_pair1__fJ = rows[_CABLC_CHANNEL] + rows["dswct"] + rows[_ARRAY_ROW]
    fitted_pair2__fJ = rows["sinwp_sc"] + rows["pn_isub"]
    emit(
        f"Pair checks: CABLC+DSWCT {fitted_pair1__fJ:.4f}/{pair1_target__fJ:.4f} fJ/access; "
        f"SINWP-SC+PN-ISUB {fitted_pair2__fJ:.4f}/{pair2_target__fJ:.4f} fJ/access."
    )
    fixed_tmcsa__fJ = config.tmcsa_config.e_per_step__fJ * config.adc_config.bits * io_num
    alpha_raw = fitted_conduction_scale(
        rows,
        target__fJ=tmcsa_target__fJ,
        fixed__fJ=fixed_tmcsa__fJ,
        current_scale=config.tmcsa_config.conduction_scale,
    )
    alpha = round_significant(alpha_raw)
    config = set_conduction_scale(config, alpha)
    emit(
        f"Fixed switching {fixed_tmcsa__fJ:.4f} fJ/access "
        f"({config.adc_config.bits} steps x {io_num} IO x {config.tmcsa_config.e_per_step__fJ:.1f} fJ); "
        f"PH2/PH3 durations remain {config.tmcsa_config.t_ph2__ns:#.5g}/"
        f"{config.tmcsa_config.t_ph3__ns:#.5g} ns; residual conduction_scale "
        f"{alpha_raw:.8g}, rounded to {alpha:#.5g}."
    )

    emit()
    emit("## Values to write back")
    emit()
    emit(f"- `reference_config.i_refs__uA = [[{', '.join(f'{value:#.5g}' for value in ladder)}]]`")
    emit(f"- `pn_isub_config.e_per_op__fJ = {config.pn_isub_config.e_per_op__fJ:#.5g}`")
    emit(f"- `tmcsa_config.e_per_step__fJ = {config.tmcsa_config.e_per_step__fJ:#.5g}`")
    emit(f"- `tmcsa_config.conduction_scale = {config.tmcsa_config.conduction_scale:#.5g}`")

    args.report.write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
