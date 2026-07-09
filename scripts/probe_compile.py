"""Cold-compile probe for the physical xbar macro — measures compile time,
warm runtime, and peak compiled-mode memory under realistic leading-batch
sizes.

Builds a minimal ``DirectCimUnit`` wrapping an ``Offset1T1RCimMacro`` from the
1t1r_28nm preset, then triggers one cold ``@torch.compile`` of the macro
``matmul`` followed by one warm call. Reports:

  * cold wall (compile + run)
  * warm wall (run only)
  * inferred compile time (cold − warm)
  * peak CUDA memory after the warm call (``max_memory_allocated``) — this
    is what production cares about, because it tells you whether a given
    block-tri solver implementation can survive the compiled forward at
    LeNet- or BERT-sized leading batches.

The ``--solver`` flag picks which block-tridiagonal implementation
``neurox.primitive.xbar.solver.nested`` will call. The probe monkey-patches
the symbol at import time, so the same script can sweep Thomas / PCR /
dense without manual edits.

Outputs are mirrored to ``log/probe_compile/<tag>.log``. Pair with
``TORCH_LOGS="+inductor"`` to capture the inductor profile in the same
log.

Typical realistic sweep (mirrors LeNet conv1's M ≈ 784 leading batch)::

    for s in thomas pcr dense; do
      python scripts/probe_compile.py \\
          --solver $s --row-num 64 --col-num 64 \\
          --batch-size 8 --seq-len 784 --tag realistic_${s}_row64
    done
"""

# ruff: noqa: T201

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import IO

import torch

import neurox.primitive.xbar.solver._linalg as _linalg

# IMPORTANT: monkey-patch the solver before importing anything that pulls
# in the nested solver, otherwise the resolved binding sticks.
import neurox.primitive.xbar.solver.nested as _nested

_SOLVER_TABLE = {
    "thomas": _linalg.solve_block_tridiagonal,
    "pcr": _linalg.solve_block_tridiagonal_pcr,
    "dense": _linalg.solve_block_tridiagonal_dense,
}


def _bind_solver(name: str) -> None:
    if name not in _SOLVER_TABLE:
        raise SystemExit(f"unknown --solver {name!r}; pick from {sorted(_SOLVER_TABLE)}")
    _nested.solve_block_tridiagonal = _SOLVER_TABLE[name]


import works.offset_1t1r  # noqa: F401  (register the Offset1T1RCimMacro kind)
from neurox.architecture.unit.cim import (
    CimUnit,
    DirectCimUnitConfig,
    DirectCimUnitPolicy,
)
from neurox.common import dataclass_from_file
from neurox.primitive.physical_constant import T_ROOM__K
from neurox.primitive.analog import SwitchCapPolicy, VoltageDriverPolicy, VoltageMuxPolicy
from neurox.primitive.analog.adc import AdcOperationPoint, McsSarAdcConfig, McsSarAdcPolicy
from neurox.primitive.analog.dac import GeneralDACPolicy
from neurox.primitive.analog.tia import OpAmpTIAPolicy
from neurox.primitive.analog.voltage_reference import VoltageReferencePolicy
from neurox.primitive.device import NMOSPolicy, RRAMPolicy
from neurox.primitive.digital import AccumulatorConfig
from neurox.primitive.xbar.array import XbarArray1T1RPolicy
from neurox.primitive.xbar.cell import XbarCell1T1RPolicy
from works.offset_1t1r.macro import Offset1T1RCimMacroConfig, Offset1T1RCimMacroPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"


class _Tee:
    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, msg: str) -> int:
        for s in self._streams:
            s.write(msg)
        return len(msg)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _all_off_xbar_policy(config: Offset1T1RCimMacroConfig) -> Offset1T1RCimMacroPolicy:
    if not isinstance(config.adc_config, McsSarAdcConfig):
        raise TypeError(f"probe expects McsSarAdcConfig; got {type(config.adc_config).__name__}")
    return Offset1T1RCimMacroPolicy(
        array=XbarArray1T1RPolicy(
            cell=XbarCell1T1RPolicy(
                rram=RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False),
                nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
            ),
            solve_chunk_size=0,
        ),
        wl_dac=GeneralDACPolicy(drive_thermal=False),
        tia=OpAmpTIAPolicy(
            opamp_gain_sigma=False,
            nmos=NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        ),
        sl_driver=VoltageDriverPolicy(offset=False, thermal=False),
        clamp_ref=VoltageReferencePolicy(tolerance=False, noise=False),
        signal_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
        ref_switchcap=SwitchCapPolicy(cap_mismatch=False, sampling_thermal_noise=False),
        voltage_mux=VoltageMuxPolicy(mux_gain_mismatch=False, mux_noise_cm=False, mux_noise_dm=False),
        bl_adc=McsSarAdcPolicy(
            cap_mismatch=False,
            comparator_offset=False,
            comparator_thermal_noise=False,
            sampling_thermal_noise=False,
        ),
        adc_v_ref=VoltageReferencePolicy(tolerance=False, noise=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-compile probe for physical xbar macro.matmul")
    parser.add_argument("--solver", choices=tuple(_SOLVER_TABLE), default="thomas",
                        help="Which solve_block_tridiagonal impl to bind into the Newton solver")
    parser.add_argument("--row-num", type=int, default=64)
    parser.add_argument("--col-num", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2, help="Leading batch dim on the input")
    parser.add_argument("--seq-len", type=int, default=4, help="M (sequence-like) dim on the input")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--log-dir", default=str(REPO_ROOT / "log" / "probe_compile"))
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-mode", type=int, default=0)
    parser.add_argument("--cold-timeout", type=int, default=900, help="Skip warm call if cold exceeded this many seconds")
    args = parser.parse_args()

    _bind_solver(args.solver)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.tag}.log"
    log_file = log_path.open("w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    try:
        _run(args)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"[probe] log written to {log_path}")


def _run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        print(f"[probe] device={device}  free={free / 1e9:.2f} GB / total={total / 1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats(device)

    print(f"[probe] solver={args.solver}")

    if not PRESET.is_file():
        raise SystemExit(f"missing preset: {PRESET}")

    xbar_cfg_full = dataclass_from_file(Offset1T1RCimMacroConfig, PRESET, section="cim_macro")
    ref_group_size = min(xbar_cfg_full.ref_group_size, args.col_num)
    while args.col_num % ref_group_size != 0:
        ref_group_size -= 1
    ref_location = min(xbar_cfg_full.ref_location, ref_group_size)
    xbar_cfg = replace(
        xbar_cfg_full,
        col_num=args.col_num,
        row_num=args.row_num,
        ref_group_size=ref_group_size,
        ref_location=ref_location,
    )
    print(
        f"[probe] geometry: col_num={xbar_cfg.col_num} row_num={xbar_cfg.row_num} "
        f"ref_group_size={xbar_cfg.ref_group_size} ref_location={xbar_cfg.ref_location}"
    )
    print(f"[probe] leading-batch on input: batch={args.batch_size} seq_len={args.seq_len}")

    accum_cfg = AccumulatorConfig(
        bit_width=32, energy_per_op__fJ=0.0, latency_per_op__ns=0.0,
        leakage_per_inst__uW=0.0, area_per_inst__um2=0.0,
    )
    direct_cfg = DirectCimUnitConfig(
        xbar_config=xbar_cfg,
        w_encoding="true_form",
        col_accumulator_config=accum_cfg,
    )
    policy = DirectCimUnitPolicy(cim_macro=_all_off_xbar_policy(xbar_cfg))

    n_logical = xbar_cfg.col_num
    k_logical = xbar_cfg.row_num
    macro = CimUnit.from_config(
        config=direct_cfg,
        policy=policy,
        name="probe",
        w_logical_shape=(n_logical, k_logical),
        dtype=torch.float32,
        T__K=T_ROOM__K,
        ideal_xbar=False,
    )
    macro = macro.to(device).eval()
    macro.fabricate()

    w_lo, w_hi = macro.w_value_range
    x_lo, x_hi = macro.x_value_range
    weight = torch.randint(w_lo, w_hi + 1, (n_logical, k_logical), dtype=torch.int32, device=device)
    macro.program(weight)

    input_tensor = torch.randint(
        x_lo, x_hi + 1,
        (args.batch_size, args.seq_len, k_logical),
        dtype=torch.int32, device=device,
    )
    print(f"[probe] input shape={tuple(input_tensor.shape)}")

    op = AdcOperationPoint(adc_mode=args.adc_mode, adc_bits=args.adc_bits)

    print("[probe] starting COLD matmul (triggers @torch.compile) ...")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    out_cold = macro.matmul(input_tensor, adc_operation_point=op)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_cold = time.time() - t0
    peak_cold = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    print(f"[probe] COLD: wall={t_cold:.2f}s  peak_mem={peak_cold:.3f} GB  out_shape={tuple(out_cold.shape)}")

    print("[probe] starting WARM matmul (run only) ...")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t1 = time.time()
    out_warm = macro.matmul(input_tensor, adc_operation_point=op)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_warm = time.time() - t1
    peak_warm = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    print(f"[probe] WARM: wall={t_warm:.4f}s  peak_mem={peak_warm:.3f} GB  out_sum={out_warm.sum().item()}")

    print("[probe] -----------------------------------------------")
    print(f"[probe] SUMMARY  solver={args.solver}  geom=(row={args.row_num} col={args.col_num})  "
          f"batch={args.batch_size}  seq={args.seq_len}")
    print(f"[probe] cold (compile + run):  {t_cold:.2f}s    peak={peak_cold:.3f} GB")
    print(f"[probe] warm (run only):       {t_warm:.4f}s   peak={peak_warm:.3f} GB")
    print(f"[probe] inferred compile time: {t_cold - t_warm:.2f}s")


if __name__ == "__main__":
    main()
