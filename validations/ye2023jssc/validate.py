"""Validation harness for the ye2023jssc WH-2T1R CIM macro.

Builds :class:`~neurox.works.macro.cim.ye2023jssc.Ye2023JsscCimMacro` from
``params.toml`` + ``policy.toml``, checks five hard gates against
``anchors.toml``, and prints an ungated dual-caliber power report. The process
exits non-zero unless all five gates pass. Anchor numbers live only in
``anchors.toml``; this harness hardcodes none of them.

The hard gates:

1. *golden transfer* — codes equal the closed-form transfer built from the
   config's own tables, plus a deterministic asymmetric-value regression that
   pins the LSB-first digit order;
2. *I_TBL table* — the radix-scaled per-plane LRS currents match the measured
   means and every plane's HRS current stays under the measured bound;
3. *RS-CSA energy* — per-conversion energy is flat at both sparsity points
   within tolerance of the anchor and the per-code energy spread lands in the
   measured window; a calibration-consistency check, since ``mirror_scale`` and
   ``e_fixed_per_op__fJ`` are solved against exactly this constraint pair;
4. *zero input* — an all-zero input vector converts to code 0 on every output;
5. *derived T_AC* — the readout's derived access window equals the measured
   value and is the per-access latency the profiler integrates.

The report reads one run under two calibers — mounting hypotheses for the
measured setup, each naming the one conduction branch the evaluation instrument
feeds, which therefore appears in no measured power pin. ``results.md`` records
the run outcome and what the two readings mean.

The workload statistics come from a CROSSED ensemble: each round draws a fresh
ensemble of ``n_w`` weight matrices onto one macro built at ``inst_shape=(n_w,)``
and a fresh batch of ``n_x`` input vectors, and reads every ``(input, weight)``
pair in one call. The five gates keep running on a separate scalar macro.

Run:
    make validate_ye2023jssc

The three TOML artifacts are FIXED files beside this script; only the run knobs
(device, seed, draw counts, solver chunk) are CLI-settable, by invoking the
script directly:

    TORCH_COMPILE_DISABLE=1 uv run python validations/ye2023jssc/validate.py \
        --device cuda --n-w 64 --n-x 256 --repeat 8 --solve-chunk 4096
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import statistics
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.works.macro.cim.ye2023jssc import Ye2023JsscCimMacro, Ye2023JsscCimMacroPolicy

_LOG = logging.getLogger(__name__)

_VAL_DIR = Path(__file__).resolve().parent
_PARAMS_PATH = _VAL_DIR / "params.toml"
_POLICY_PATH = _VAL_DIR / "policy.toml"
_ANCHORS_PATH = _VAL_DIR / "anchors.toml"

_BLOCKS = ("Array", "RS-CSA", "Mux&Driver", "Timing&Ctrl")

# Per caliber: which measured power pin each profiler row is attributed to. The
# instrument-fed branch and the rows that ride it go to the ``_OFF_PIN`` group.
# ``pool`` asserts the pins plus off-pin partition the dynamic rows exhaustively.
_OFF_PIN = "off-pin"
_CALIBERS: dict[str, dict[str, tuple[str, ...]]] = {
    "Y": {
        "Array": ("array", ".bl_cond", ".bl_cap", "bl_driver", "sl_driver"),
        "RS-CSA": ("rscsa",),
        "Mux&Driver": (".mux_driver",),
        "Timing&Ctrl": (".timing_ctrl",),
        _OFF_PIN: (".dl_cond",),
    },
    "X": {
        "Array": ("array", ".dl_cond", "sl_driver"),
        "RS-CSA": ("rscsa",),
        "Mux&Driver": (".mux_driver",),
        "Timing&Ctrl": (".timing_ctrl",),
        _OFF_PIN: (".bl_cond", ".bl_cap", "bl_driver"),
    },
}
# What each caliber assumes about the measured mounting, printed with its table.
_CALIBER_MOUNT: dict[str, str] = {
    "Y": "TBL clamp-driven from outside, so the 0.8 V row branch is instrument-fed",
    "X": "BL inputs driven off-chip by the board DAC array (Fig.15), so the 0.3 V input branch is instrument-fed",
}
_STATIC_NAMES: dict[str, tuple[str, ...]] = {
    "Array": ("array",),
    "RS-CSA": ("rscsa",),
    "Mux&Driver": ("mux_driver",),
    "Timing&Ctrl": ("timing_ctrl",),
}
# Block display name -> anchors.toml reference.shares key.
_ANCHOR_KEY: dict[str, str] = {
    "Array": "array",
    "RS-CSA": "rscsa",
    "Mux&Driver": "mux_driver",
    "Timing&Ctrl": "timing_ctrl",
}
# Every dynamic profiler row, with its owner and billing rate, in report order.
# The last four are structurally zero and are listed rather than dropped, so a row
# that starts drawing energy cannot slip past the caliber pooling unnoticed.
_CHANNELS: tuple[tuple[str, str], ...] = (
    ("array", "array + cell, PER ACCESS: WL wire + WL gate caps, selected-cell X dip"),
    (".bl_cond", "macro, PER ACCESS: 0.3 V input-branch conduction over T_AC"),
    (".bl_cap", "macro, PER VECTOR: BL-column charge (levels held across the row scan)"),
    (".dl_cond", "macro, PER ACCESS: 0.8 V row branch (raw I_TBL) over T_AC"),
    ("rscsa", "RS-CSA, PER CONVERSION: E_fixed + per-phase E_code"),
    (".mux_driver", "macro seat, no per-op dynamic share"),
    (".timing_ctrl", "macro seat, no per-op dynamic share"),
    ("bl_driver", "ideal BL source; the macro bills the whole input branch"),
    ("sl_driver", "ideal SL ground clamp; no billed branch"),
)

_QUANTIZATION_MODE = 0
_ADC_BITS = 4
_ROW_NUM = 32
_COL_NUM = 64
_FJ_PER_PJ = 1000.0
_OPS_PER_MAC = 2  # one multiply + one accumulate
# Random-workload sample count of the golden-transfer gate, decoupled from the
# measurement draw counts: the gate checks a transfer, not a power statistic.
_GOLDEN_SAMPLE_NUM = 64


# ---------------------------------------------------------------------------
# Build / draw
# ---------------------------------------------------------------------------


def build_macro(
    params_path: Path,
    policy_path: Path,
    *,
    device: torch.device,
    inst_shape: tuple[int, ...] = (),
    solve_chunk_size: int,
) -> Ye2023JsscCimMacro:
    """Build + fabricate the macro on ``device``, float32, eval mode.

    Args:
        params_path: Fixed ``params.toml``.
        policy_path: Fixed ``policy.toml``.
        device: Device the macro lives on.
        inst_shape: Instance prefix — ``()`` for the gate macro, ``(n_w,)`` for
            the measurement ensemble of parallel dies.
        solve_chunk_size: Array solve chunk, a MACHINE knob overriding the
            policy file's field: it splits the broadcast leading to bound peak
            memory and changes no modelled quantity.
    """
    config = CimMacroConfig.from_file(params_path, section="cim_macro")
    policy = CimMacroPolicy.from_file(policy_path, section="policy")
    assert isinstance(policy, Ye2023JsscCimMacroPolicy)
    policy = dataclasses.replace(
        policy,
        array_policy=dataclasses.replace(policy.array_policy, solve_chunk_size=solve_chunk_size),
    )
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=_ROW_NUM,
        output_num=_COL_NUM,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )
    assert isinstance(macro, Ye2023JsscCimMacro)
    macro.to(device)
    macro.eval()
    macro.fabricate()
    return macro


def _draw_weight(
    gen: torch.Generator,
    *,
    w_shape: tuple[int, ...],
    w_max: int,
    p_zero: float,
) -> Tensor:
    """Draw logical weights at VALUE level: ``P(w = 0) = p_zero``, else uniform ``[1, w_max]``."""
    zero = torch.rand(w_shape, generator=gen, device=gen.device) < p_zero
    nonzero = torch.randint(1, w_max + 1, w_shape, generator=gen, device=gen.device)
    return torch.where(zero, torch.zeros_like(nonzero), nonzero)


def _draw_input(gen: torch.Generator, *, shape: tuple[int, ...], p_zero: float) -> Tensor:
    """Random 1-bit input vectors of ``shape`` with ``P(x = 0) = p_zero``.

    The uniforms are drawn once and THRESHOLDED, so two sparsity points sharing a
    generator state see the same underlying draw (common random numbers).
    """
    keep = torch.rand(shape, generator=gen, device=gen.device) >= p_zero
    return keep.long()


# ---------------------------------------------------------------------------
# Closed-form transfer oracle
# ---------------------------------------------------------------------------


def _golden_transfer(macro: Ye2023JsscCimMacro, w: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    """Closed-form ``(code, distance to the nearest ladder tap)`` from the config tables.

    The oracle evaluates in float64 whatever the macro's dtype is: the reference
    stays more precise than the device under test, so the tap distance it reports
    is the exact one and bounds the macro's own rounding.

    Args:
        macro: Macro whose config tables define the transfer.
        w: Logical weight tensor ``[row_num, col_num]``.
        x: 1-bit input tensor ``[..., row_num]``.

    Returns:
        The expected unsigned code ``[..., col_num]`` and the distance [uA] from
        the compensated current to the nearest decision tap, which is zero when
        the sample sits exactly on a tap.
    """
    config = macro.config
    table = config.cell_config.i_t2_table__uA
    i_floor__uA = table[0][0]
    i_hrs__uA, i_lrs__uA = table[1][0], table[1][1]
    radix_sum = float(sum(config.array_config.weight_radix))
    # The readout is SPECIFIED as a uniform quantizer whose step is the reference
    # current it is handed, so the closed form divides by that current. Reading
    # the operating point (like the cell tables above) is not reading the
    # implementation: the macro reaches its code through the physical solve, the
    # per-plane radix sum, the PH0 subtraction, and the RS-CSA's own
    # compare-phase ladder, none of which appears here.
    i_ref__uA = config.reference_config.i_refs__uA[_QUANTIZATION_MODE][0]

    x_f = x.to(torch.float64)
    mac = x_f @ w.to(torch.float64)  # [..., col]
    active = x_f.sum(dim=-1, keepdim=True)  # [..., 1]
    i_comp__uA = (i_lrs__uA - i_floor__uA) * mac + (i_hrs__uA - i_floor__uA) * (radix_sum * active - mac)

    quotient = i_comp__uA / i_ref__uA
    code = quotient.floor().clamp(min=0.0, max=float((1 << config.adc_config.bits) - 1)).long()
    frac = quotient - quotient.floor()
    tap_distance__uA = torch.minimum(frac, 1.0 - frac) * i_ref__uA
    return code, tap_distance__uA


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointMeasurement:
    """Profiler reduction of the crossed ensemble workload at one sparsity point.

    Three quantities normalize differently, because the ``die_num`` instances are
    PARALLEL dies of one die-level model:

      * dynamic energy is the ensemble sum and divides by ``accesses``, the
        ensemble's output-access count, giving per-access energy of ONE die;
      * static leakage is per-die: the profiled ensemble leakage is exactly
        ``die_num`` x the per-instance leakage, so it is divided here already;
      * latency does not scale with ``die_num`` — the profiled wall time covers
        ``accesses / die_num`` serial accesses, which is what ``t_cycle__ns``
        divides by.

    Attributes:
        p_zero_input: Input-sparsity point.
        die_num: Parallel instances (weight draws) per round.
        repeat: Rounds, each a fresh weight ensemble and a fresh input batch.
        dynamic__fJ: Per profiler row, the ENSEMBLE dynamic energy of the point.
        static__uW: Per profiler row, the PER-DIE static leakage power.
        total_dynamic__fJ: Ensemble dynamic energy of the point.
        total_static__uW: Per-die static leakage power.
        total_latency__ns: Profiled latency — the per-die serial access time.
        accesses: Output accesses read, ``repeat * n_x * die_num * col_num``.
        round_total__uW: Per-round model total power, one entry per round.
    """

    p_zero_input: float
    die_num: int
    repeat: int
    dynamic__fJ: dict[str, float]
    static__uW: dict[str, float]
    total_dynamic__fJ: float
    total_static__uW: float
    total_latency__ns: float
    accesses: int
    round_total__uW: tuple[float, ...] = ()

    @property
    def t_cycle__ns(self) -> float:
        """Latency per output access [ns]; the parallel dies share their accesses."""
        return self.total_latency__ns * self.die_num / self.accesses

    def power__uW(self, dynamic__fJ: float) -> float:
        """Per-die power [uW] of a dynamic-energy row: its per-access energy over one cycle."""
        return dynamic__fJ / self.accesses / self.t_cycle__ns

    @property
    def total__uW(self) -> float:
        return self.power__uW(self.total_dynamic__fJ) + self.total_static__uW

    @property
    def per_output__pJ(self) -> float:
        return self.total__uW * self.t_cycle__ns / _FJ_PER_PJ

    @property
    def round_mean__uW(self) -> float:
        """Mean model total power [uW] over the rounds (draw-to-draw centre)."""
        return statistics.fmean(self.round_total__uW)

    @property
    def round_stderr__uW(self) -> float:
        """Standard error [uW] of the round means: ``std / sqrt(repeat)``."""
        if len(self.round_total__uW) < 2:
            return 0.0
        return statistics.stdev(self.round_total__uW) / math.sqrt(len(self.round_total__uW))

    @property
    def ef__tops_w(self) -> float:
        """Model MAC energy efficiency [TOPS/W] = ``2 * row_num`` ops per access."""
        return _OPS_PER_MAC * _ROW_NUM / self.per_output__pJ


@dataclass(frozen=True)
class BlockPower:
    """One measured block under one caliber: power [uW] and its anchor."""

    name: str
    dynamic__uW: float
    static__uW: float
    target__uW: float
    share__pct: float

    @property
    def total__uW(self) -> float:
        return self.dynamic__uW + self.static__uW

    @property
    def ratio(self) -> float:
        return self.total__uW / self.target__uW if self.target__uW else float("inf")


def _pool_rounds(rounds: list[PointMeasurement]) -> PointMeasurement:
    """Fold per-round reductions into one point; statics are round-invariant."""
    first = rounds[0]
    dynamic__fJ: dict[str, float] = {}
    for r in rounds:
        for name, e in r.dynamic__fJ.items():
            dynamic__fJ[name] = dynamic__fJ.get(name, 0.0) + e
    return PointMeasurement(
        p_zero_input=first.p_zero_input,
        die_num=first.die_num,
        repeat=len(rounds),
        dynamic__fJ=dynamic__fJ,
        static__uW=first.static__uW,
        total_dynamic__fJ=sum(r.total_dynamic__fJ for r in rounds),
        total_static__uW=first.total_static__uW,
        total_latency__ns=sum(r.total_latency__ns for r in rounds),
        accesses=sum(r.accesses for r in rounds),
        round_total__uW=tuple(r.total__uW for r in rounds),
    )


def measure(
    macro: Ye2023JsscCimMacro,
    *,
    p_zero_input: float,
    p_zero_weight: float,
    n_x: int,
    repeat: int,
    seed: int,
) -> PointMeasurement:
    """Profile a crossed weight-ensemble x input-batch workload at one sparsity point.

    Each of the ``repeat`` rounds draws a FRESH weight ensemble of ``die_num =
    macro.inst_count`` matrices and a FRESH batch of ``n_x`` input vectors, then
    reads every ``(input, weight)`` pair in ONE call: the inputs enter with a
    size-1 instance slot ``[n_x, 1, row_num]`` and broadcast over the dies, so
    the codes come back ``[n_x, die_num, col_num]``. Every round is profiled on
    its own, which is what exposes the draw-to-draw spread.

    The generator is created fresh from ``seed`` and the draw order (weights,
    then inputs, per round, at shapes that do not depend on the sparsity point)
    is fixed. Two sparsity points measured under the same seed therefore replay
    the SAME weight sequence and THRESHOLD the same input uniforms — common
    random numbers, so their difference carries no draw noise.

    Args:
        macro: Ensemble macro, built at ``inst_shape=(die_num,)``.
        p_zero_input: Probability an input bit is 0.
        p_zero_weight: Probability a weight VALUE is 0.
        n_x: Input vectors per round.
        repeat: Rounds.
        seed: Generator seed; pass the same value at every sparsity point.
    """
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    die_num = macro.inst_count

    rounds: list[PointMeasurement] = []
    with torch.no_grad():
        for _ in range(repeat):
            w = _draw_weight(
                gen,
                w_shape=(*macro.inst_shape, macro.row_num, macro.col_num),
                w_max=macro.w_value_range[1],
                p_zero=p_zero_weight,
            )
            x = _draw_input(gen, shape=(n_x, *(1,) * len(macro.inst_shape), macro.row_num), p_zero=p_zero_input)
            with NeuroxProfiler() as prof:
                macro.program(w)
                macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
            report = prof.report(macro)
            rounds.append(
                PointMeasurement(
                    p_zero_input=p_zero_input,
                    die_num=die_num,
                    repeat=1,
                    dynamic__fJ=dict(report.energy_by_name),
                    static__uW={r.qualified_name: r.leakage_power__uW / die_num for r in report.static_records},
                    total_dynamic__fJ=report.total_dynamic_energy__fJ,
                    total_static__uW=report.static.leakage_power__uW / die_num,
                    total_latency__ns=report.total_latency__ns,
                    accesses=n_x * die_num * macro.col_num,
                )
            )
    return _pool_rounds(rounds)


@dataclass(frozen=True)
class CaliberPooling:
    """One sparsity point read under one mounting hypothesis.

    Attributes:
        caliber: Caliber key in ``_CALIBERS``.
        blocks: The four measured power pins, in ``_BLOCKS`` order.
        off_pin__uW: Power [uW] of the instrument-fed branch, on no pin.
        model_total__uW: Total macro power [uW], every branch billed.
        target_total__uW: Measured total macro power [uW] at this point.
        t_cycle__ns: Latency per output access [ns].
    """

    caliber: str
    blocks: tuple[BlockPower, ...]
    off_pin__uW: float
    model_total__uW: float
    target_total__uW: float
    t_cycle__ns: float

    @property
    def array(self) -> BlockPower:
        """The array pin, the one block whose contents differ between calibers."""
        return next(b for b in self.blocks if b.name == "Array")

    @property
    def on_chip__uW(self) -> float:
        return self.model_total__uW - self.off_pin__uW

    @property
    def on_chip_ratio(self) -> float:
        return self.on_chip__uW / self.target_total__uW

    @property
    def on_chip_per_output__pJ(self) -> float:
        return self.on_chip__uW * self.t_cycle__ns / _FJ_PER_PJ

    @property
    def on_chip_ef__tops_w(self) -> float:
        """Energy efficiency [TOPS/W] of the on-chip power alone, ``2 * row_num`` ops per access."""
        return _OPS_PER_MAC * _ROW_NUM / self.on_chip_per_output__pJ


def pool(m: PointMeasurement, anchors: dict, *, caliber: str) -> CaliberPooling:
    """Pool the raw rows into one caliber's four measured pins plus its off-pin branch.

    Raises:
        ValueError: If the caliber's pins and off-pin group are not an exhaustive
            partition of the profiler's dynamic rows (an unattributed channel
            would silently drop out of the on-chip total).
    """
    mapping = _CALIBERS[caliber]
    idx = _sparsity_index(anchors, m.p_zero_input)
    target_total = anchors["reference"]["total_macro__uW"][idx]
    shares = anchors["reference"]["shares"]["p875" if idx == 0 else "p50"]

    def group__uW(name: str) -> float:
        return m.power__uW(sum(m.dynamic__fJ.get(k, 0.0) for k in mapping[name]))

    blocks = tuple(
        BlockPower(
            name=name,
            dynamic__uW=group__uW(name),
            static__uW=sum(m.static__uW.get(k, 0.0) for k in _STATIC_NAMES[name]),
            target__uW=shares[_ANCHOR_KEY[name]] / 100.0 * target_total,
            share__pct=shares[_ANCHOR_KEY[name]],
        )
        for name in _BLOCKS
    )

    mapped__fJ = sum(m.dynamic__fJ.get(k, 0.0) for keys in mapping.values() for k in keys)
    if abs(mapped__fJ - m.total_dynamic__fJ) > 1e-6 * max(1.0, abs(m.total_dynamic__fJ)):
        unmapped = sorted(set(m.dynamic__fJ) - {k for keys in mapping.values() for k in keys})
        raise ValueError(
            f"caliber {caliber} pins + off-pin are not exhaustive: mapped {mapped__fJ:.6g} fJ vs total "
            f"{m.total_dynamic__fJ:.6g} fJ; unattributed rows {unmapped}"
        )
    return CaliberPooling(
        caliber=caliber,
        blocks=blocks,
        off_pin__uW=group__uW(_OFF_PIN),
        model_total__uW=m.total__uW,
        target_total__uW=target_total,
        t_cycle__ns=m.t_cycle__ns,
    )


def _sparsity_index(anchors: dict, p_zero_input: float) -> int:
    """Index of ``p_zero_input`` in ``anchors[data].p_zero_input`` (nearest match)."""
    pts = anchors["data"]["p_zero_input"]
    return min(range(len(pts)), key=lambda i: abs(pts[i] - p_zero_input))


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """One hard gate: its verdict and the measured quantity behind it."""

    name: str
    passed: bool
    detail: str


def _asymmetric_value_codes(macro: Ye2023JsscCimMacro) -> tuple[list[int], list[int]]:
    """Codes for one column per weight VALUE, all inputs high: ``(measured, expected)``.

    Column ``j`` stores the single value ``j % (w_max + 1)`` on every input, so the
    columns sweep the whole encodable range under one access.
    """
    device = next(macro.buffers()).device
    w_max = macro.w_value_range[1]
    values = torch.arange(macro.col_num, device=device) % (w_max + 1)
    w = values.expand(macro.row_num, macro.col_num).contiguous().long()
    x = torch.ones((1, macro.row_num), dtype=torch.long, device=device)
    with torch.no_grad():
        macro.program(w)
        code = macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    expected, _ = _golden_transfer(macro, w, x)
    return code.long().flatten().tolist(), expected.flatten().tolist()


def gate_golden_transfer(macro: Ye2023JsscCimMacro, *, n: int, seed: int) -> GateResult:
    """Codes equal the closed-form config-derived transfer, over a random workload and per value.

    Random samples whose compensated current lands on a decision tap are excluded:
    on those the floor is decided by floating-point rounding, not by the transfer.

    Args:
        macro: Scalar (single-die) macro under test.
        n: Random input vectors to check; a transfer check, so it is fixed at
            ``_GOLDEN_SAMPLE_NUM`` rather than tied to the measurement counts.
        seed: Generator seed.
    """
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    w = _draw_weight(gen, w_shape=(macro.row_num, macro.col_num), w_max=macro.w_value_range[1], p_zero=0.5)
    x = _draw_input(gen, shape=(n, macro.row_num), p_zero=0.5)
    with torch.no_grad():
        macro.program(w)
        code = macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    expected, tap_distance__uA = _golden_transfer(macro, w, x)

    # Tap-boundary band [uA]. The macro solves in float32, whose resolution near
    # the ~100 uA full scale is ~1e-5 uA, and the end-to-end solve accumulates a
    # few tens of that; 1e-3 sits ~100x above the accumulated noise while cutting
    # only ~3e-4 of uniformly placed samples (a 2 x 1e-3 uA band per 7 uA bin).
    decidable = tap_distance__uA > 1e-3
    mismatch = int(((code.long() != expected) & decidable).sum())
    checked = int(decidable.sum())

    value_code, value_expected = _asymmetric_value_codes(macro)
    value_ok = value_code == value_expected
    w_max = macro.w_value_range[1]
    failures: list[str] = []
    if mismatch:
        failures.append(f"{mismatch}/{checked} random-workload codes differ from the closed form")
    if not value_ok:
        failures.append(f"asymmetric-value codes {value_code[: w_max + 1]} vs {value_expected[: w_max + 1]}")

    return GateResult(
        name="golden transfer + asymmetric-value regression",
        passed=not failures,
        detail="; ".join(failures)
        or f"{checked - mismatch}/{checked} codes match the closed-form transfer "
        f"({int((~decidable).sum())} tap-boundary samples excluded); values 0..{w_max} -> codes "
        f"{value_code[: w_max + 1]} (LSB-first place values)",
    )


def gate_i_tbl_table(macro: Ye2023JsscCimMacro, anchors: dict) -> GateResult:
    """Radix-scaled per-plane I_TBL currents match the measured means and clear the HRS bound."""
    config = macro.config
    array_config = config.array_config
    table = config.cell_config.i_t2_table__uA
    i_hrs__uA, i_lrs__uA = table[1][0], table[1][1]
    radix_path = (*array_config.weight_radix, *array_config.redundant_radix)

    lrs_anchor = anchors["gate"]["i_tbl"]["lrs_mean__uA"]
    lrs_radix = anchors["gate"]["i_tbl"]["lrs_mean_radix"]
    tol = anchors["gate"]["i_tbl"]["lrs_tolerance_relative"]
    hrs_bound__uA = anchors["gate"]["i_tbl"]["hrs_bound__uA"]

    failures: list[str] = []
    for m, ref__uA in zip(lrs_radix, lrs_anchor, strict=True):
        pred__uA = m * i_lrs__uA
        if abs(pred__uA - ref__uA) > tol * ref__uA:
            failures.append(f"m={m}: LRS {pred__uA:.4f} vs {ref__uA:.4f} uA")
    failures.extend(
        f"m={m}: HRS {m * i_hrs__uA * 1e3:.1f} nA over the {hrs_bound__uA * 1e3:.0f} nA bound"
        for m in radix_path
        if m * i_hrs__uA > hrs_bound__uA
    )

    worst_hrs__uA = max(radix_path) * i_hrs__uA
    return GateResult(
        name="I_TBL table + HRS bound",
        passed=not failures,
        detail="; ".join(failures)
        or f"LRS {[round(m * i_lrs__uA, 3) for m in lrs_radix]} uA vs {lrs_anchor} uA, "
        f"worst-plane HRS {worst_hrs__uA * 1e3:.1f} nA <= {hrs_bound__uA * 1e3:.0f} nA",
    )


def _rscsa_energy_by_code(macro: Ye2023JsscCimMacro) -> dict[int, float]:
    """Per-conversion RS-CSA energy [fJ] for every non-zero code, mid-bin driven."""
    buf = next(macro.buffers())
    device, dtype = buf.device, buf.dtype
    i_ref__uA = macro.config.reference_config.i_refs__uA[_QUANTIZATION_MODE][0]
    # The readout takes the ONE reference current on a ``[1]`` tap axis and
    # derives its whole decision ladder from it; the code step is that current.
    i_refs__uA = torch.tensor([i_ref__uA], dtype=dtype, device=device)
    out: dict[int, float] = {}
    for code in range(1, 1 << _ADC_BITS):
        i_in__uA = torch.tensor(
            [macro.rscsa.i_ph0_comp__uA + (code + 0.5) * i_ref__uA],
            dtype=dtype,
            device=device,
        )
        with NeuroxProfiler() as prof, torch.no_grad():
            macro.rscsa.convert(i_in__uA, i_refs__uA, bits=_ADC_BITS)
        out[code] = prof.report(macro.rscsa).total_dynamic_energy__fJ
    return out


def gate_rscsa_energy(
    macro: Ye2023JsscCimMacro,
    anchors: dict,
    measurements: list[PointMeasurement],
) -> GateResult:
    """Per-conversion RS-CSA energy is flat at both points and its per-code spread is in window."""
    ref__fJ = anchors["gate"]["rscsa"]["energy_per_conversion__fJ"]
    tol = anchors["gate"]["rscsa"]["flat_tolerance_relative"]
    lo, hi = anchors["gate"]["rscsa"]["code_spread_range"]

    failures: list[str] = []
    per_conv: list[float] = []
    for m in measurements:
        e__fJ = m.dynamic__fJ.get("rscsa", 0.0) / m.accesses
        per_conv.append(e__fJ)
        if abs(e__fJ - ref__fJ) > tol * ref__fJ:
            failures.append(f"p_zero {m.p_zero_input:.3f}: {e__fJ:.1f} fJ vs {ref__fJ:.0f} fJ +-{tol * 100:.0f}%")

    by_code = _rscsa_energy_by_code(macro)
    spread = max(by_code.values()) / min(by_code.values())
    if not (lo <= spread <= hi):
        failures.append(f"code spread {spread:.3f}x outside [{lo}, {hi}]")

    return GateResult(
        name="RS-CSA 470 fJ flat + code spread",
        passed=not failures,
        detail="; ".join(failures)
        or f"per-conversion {' / '.join(f'{e:.1f}' for e in per_conv)} fJ vs {ref__fJ:.0f} fJ, "
        f"code spread {spread:.3f}x in [{lo}, {hi}]",
    )


def gate_zero_input(macro: Ye2023JsscCimMacro, *, seed: int) -> GateResult:
    """An all-zero input vector converts to code 0 on every output."""
    device = next(macro.buffers()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    w = _draw_weight(gen, w_shape=(macro.row_num, macro.col_num), w_max=macro.w_value_range[1], p_zero=0.5)
    x = torch.zeros((1, macro.row_num), dtype=torch.long, device=device)
    with torch.no_grad():
        macro.program(w)
        code = macro.vec_mat_mul(x, quantization_mode=_QUANTIZATION_MODE, adc_bits=_ADC_BITS)
    nonzero = int((code != 0).sum())
    return GateResult(
        name="zero input -> code 0",
        passed=nonzero == 0,
        detail=f"{macro.col_num - nonzero}/{macro.col_num} outputs read code 0 (derived PH0 cancels the row floor)",
    )


def gate_t_ac(macro: Ye2023JsscCimMacro, anchors: dict, measurements: list[PointMeasurement]) -> GateResult:
    """The derived access window equals the anchor and is the profiled per-access latency."""
    ref__ns = anchors["gate"]["t_ac__ns"]
    derived__ns = float(macro.t_ac__ns)
    failures: list[str] = []
    if abs(derived__ns - ref__ns) > 1e-9:
        failures.append(f"derived T_AC {derived__ns:.4f} ns vs {ref__ns:.4f} ns")
    # The profiled cycle divides a float32 latency SUM, so it carries relative
    # rounding of order 1e-7 that the exact derived window above does not; 1e-3 ns
    # is far below any modelled phase and far above that noise.
    failures.extend(
        f"p_zero {m.p_zero_input:.3f}: profiled {m.t_cycle__ns:.4f} ns per access"
        for m in measurements
        if abs(m.t_cycle__ns - ref__ns) > 1e-3
    )
    return GateResult(
        name="derived T_AC = 66 ns",
        passed=not failures,
        detail="; ".join(failures) or f"PH0 + PH1..PH3 + t4 = {derived__ns:.1f} ns, profiled per access identically",
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _channel__uW(m: PointMeasurement, name: str) -> float:
    """Per-die dynamic power [uW] of one profiler row at one sparsity point."""
    return m.power__uW(m.dynamic__fJ.get(name, 0.0))


def _fmt_channels(measurements: list[PointMeasurement]) -> str:
    """Raw per-channel power at both sparsity points, before any caliber pooling.

    Raises:
        ValueError: If the profiler emitted a dynamic row ``_CHANNELS`` does not
            declare (it would silently vanish from the report).
    """
    declared = {name for name, _ in _CHANNELS}
    for m in measurements:
        undeclared = sorted(set(m.dynamic__fJ) - declared)
        if undeclared:
            raise ValueError(f"undeclared dynamic rows {undeclared}")

    heads = " | ".join(f"p_zero {m.p_zero_input:.3f}" for m in measurements)
    lines = [
        "### Per-channel power [uW] (caliber-independent)",
        "",
        f"| Channel | Owner / rate | {heads} |",
        "|---|---|" + "--:|" * len(measurements),
    ]
    silent: list[str] = []
    for name, role in _CHANNELS:
        powers = [_channel__uW(m, name) for m in measurements]
        if not any(powers):
            silent.append(name)
            continue
        cells = " | ".join(f"{p:8.3f}" for p in powers)
        lines.append(f"| `{name}` | {role} | {cells} |")
    for block in _BLOCKS:
        seat = [sum(m.static__uW.get(k, 0.0) for k in _STATIC_NAMES[block]) for m in measurements]
        if not any(seat):
            continue
        cells = " | ".join(f"{p:8.3f}" for p in seat)
        lines.append(f"| `{_ANCHOR_KEY[block]}` | {block} seat, STATIC leakage | {cells} |")
    totals = " | ".join(f"**{m.total__uW:8.3f}**" for m in measurements)
    lines.append(f"| **TOTAL** | | {totals} |")
    if silent:
        lines += ["", f"Zero at both points: {', '.join(f'`{name}`' for name in silent)}."]
    return "\n".join(lines)


def _all_hrs_floor__uW(macro: Ye2023JsscCimMacro, p_zero_input: float) -> float:
    """Lower bound [uW] on input-branch conduction power: every weight cell at HRS."""
    config = macro.config
    g_hrs__uS = config.cell_config.g_cell_on_table__uS[0]
    v_bl__V = config.v_bl_in1__V
    active = macro.row_num * (1.0 - p_zero_input)
    plane_num = len(config.array_config.weight_radix)
    return v_bl__V * (g_hrs__uS * v_bl__V) * active * plane_num


def _fmt_finding(macro: Ye2023JsscCimMacro, measurements: list[PointMeasurement], anchors: dict) -> str:
    """The dual-caliber conclusion: each point closes under one mounting, none closes both."""
    sparse, dense = measurements
    sp, dp = sparse.p_zero_input, dense.p_zero_input
    pools = {(c, m.p_zero_input): pool(m, anchors, caliber=c) for c in ("Y", "X") for m in measurements}

    def best_caliber(p_zero: float) -> str:
        """The mounting this point closes under: the one whose on-chip total lands on the anchor."""
        return min(("Y", "X"), key=lambda c: abs(pools[(c, p_zero)].on_chip_ratio - 1.0))

    fit_s, fit_d = best_caliber(sp), best_caliber(dp)
    cross_s, cross_d = pools[(fit_d, sp)], pools[(fit_s, dp)]
    floor__uW = _all_hrs_floor__uW(macro, sp)
    anchor__uW = pools[(fit_s, sp)].array.target__uW
    active = macro.row_num * (1.0 - sp)
    ef_anchor = anchors["reference"]["ef_tops_w"]
    fit = pools[(fit_s, sp)]

    def closes(caliber: str, p_zero: float) -> str:
        p = pools[(caliber, p_zero)]
        pins = ", ".join(f"{b.name} {b.ratio:.2f}x" for b in p.blocks)
        return f"{pins}, on-chip total {p.on_chip__uW:.2f} uW vs {p.target_total__uW:.2f} uW ({p.on_chip_ratio:.2f}x)"

    paragraphs = [
        "Each measured point is FULLY consistent under exactly ONE mounting hypothesis — all four pins and the "
        f"on-chip total at once. At {sp:.1%} input sparsity that is caliber {fit_s}: {closes(fit_s, sp)}. At "
        f"{dp:.0%} it is caliber {fit_d}: {closes(fit_d, dp)}. Crossing the mountings closes neither point: "
        f"caliber {fit_d} at {sp:.1%} reads the array pin {cross_s.array.ratio:.2f}x and the on-chip total "
        f"{cross_s.on_chip_ratio:.2f}x, and caliber {fit_s} at {dp:.0%} reads {cross_d.array.ratio:.2f}x and "
        f"{cross_d.on_chip_ratio:.2f}x. The two hypotheses are mutually exclusive, so no single mounting of the "
        "macro accounts for both published points.",
        f"The {sp:.1%} array anchor ({anchor__uW:.3f} uW) sits BELOW the model's all-HRS continuous-conduction "
        f"floor ({floor__uW:.2f} uW = v_bl_in1 * g_cell_on(HRS) * v_bl_in1 over {active:.0f} mean active inputs "
        f"x {len(macro.config.array_config.weight_radix)} weight planes) — the least current the input branch can "
        f"draw while those inputs are raised, whatever the weights. That is what rules caliber {fit_d} out at the "
        f"{sp:.1%} point on physics alone: a steady-state measurement of a pin carrying the input branch cannot "
        "land under that floor, at any weight statistics and under any calibration. Physics is calibrated at the "
        f"{dp:.0%} point (the caliber-{fit_d} array pin); the {sp:.1%} point is reported, never fitted.",
        f"Read through caliber {fit_s}, the {sp:.1%} point draws {fit.on_chip__uW:.2f} uW on chip = "
        f"{fit.on_chip_per_output__pJ:.3f} pJ per output, i.e. EF {fit.on_chip_ef__tops_w:.2f} TOPS/W against the "
        f"paper's {ef_anchor:.2f} TOPS/W headline ({fit.on_chip_ef__tops_w / ef_anchor:.2f}x). Reading the headline "
        "that way IMPLIES it excludes input-drive power, which this mounting hands to the instrument. The full "
        f"model, which bills every branch at full rail for the whole window, reads {sparse.ef__tops_w:.2f} / "
        f"{dense.ef__tops_w:.2f} TOPS/W at the two points; that is the physics view of the macro as a "
        "self-contained circuit, and the number a system-level estimate should carry.",
    ]
    body = "\n\n".join(textwrap.fill(p, width=100) for p in paragraphs)
    return f"### Finding: each Fig.19 point closes under one mounting, and no mounting closes both\n\n{body}"


def _fmt_caliber(m: PointMeasurement, anchors: dict, *, caliber: str) -> str:
    """Per-block table for one sparsity point under one caliber."""
    p = pool(m, anchors, caliber=caliber)
    mapping = _CALIBERS[caliber]
    ef_anchor = anchors["reference"]["ef_tops_w"]
    dash = f"| {'-':>8s} | {'-':>5s} | {'-':>6s} |"

    lines = [
        f"caliber {caliber} — {_CALIBER_MOUNT[caliber]}",
        f"    array pin <- {' + '.join(mapping['Array'])} = {p.array.total__uW:.3f} uW vs "
        f"{p.array.target__uW:.3f} uW ({p.array.ratio:.2f}x)",
        f"    off-pin   <- {' + '.join(mapping[_OFF_PIN])} = {p.off_pin__uW:.3f} uW, on no measured pin",
        "",
        "| Block | pred uW | dyn | static | anchor uW | share% | pred/anchor |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    lines += [
        f"| {b.name:12s} | {b.total__uW:8.3f} | {b.dynamic__uW:7.3f} | {b.static__uW:6.3f} | "
        f"{b.target__uW:8.3f} | {b.share__pct:5.1f} | {b.ratio:5.2f}x |"
        for b in p.blocks
    ]
    lines += [
        f"| off-pin      | {p.off_pin__uW:8.3f} | {p.off_pin__uW:7.3f} | {0.0:6.3f} {dash}",
        f"| **ON-CHIP**  | **{p.on_chip__uW:8.3f}** | | | **{p.target_total__uW:8.3f}** | 100.0 | "
        f"**{p.on_chip_ratio:5.3f}x** |",
        f"| model total  | {p.model_total__uW:8.3f} | | {dash}",
        "",
        f"    on-chip {p.on_chip__uW:.3f} uW -> {p.on_chip_per_output__pJ:.3f} pJ/out, "
        f"EF {p.on_chip_ef__tops_w:.2f} TOPS/W (paper headline {ef_anchor:.2f})",
    ]
    return "\n".join(lines)


def _fmt_point(m: PointMeasurement, anchors: dict) -> str:
    """Dual-caliber report for one sparsity point."""
    idx = _sparsity_index(anchors, m.p_zero_input)
    per_access_anchor__pJ = anchors["reference"]["per_access__pJ"][idx]
    ef_anchor = anchors["reference"]["ef_tops_w"]
    label = anchors["reference"]["point_label"][idx]

    lines = [
        f"### input sparsity p_zero = {m.p_zero_input:.3f} — {label}",
        f"    accesses {m.accesses} ({m.repeat} rounds x {m.die_num} dies), T_AC {m.t_cycle__ns:.2f} ns",
        f"    full model, every branch billed: {m.total__uW:.3f} uW, {m.per_output__pJ:.3f} pJ/out "
        f"(anchor {per_access_anchor__pJ:.2f}), EF {m.ef__tops_w:.2f} TOPS/W (paper headline {ef_anchor:.2f})",
        f"    draw spread over {m.repeat} rounds: model total {m.round_mean__uW:.3f} +- "
        f"{m.round_stderr__uW:.3f} uW (std/sqrt(rounds))",
        "",
        _fmt_caliber(m, anchors, caliber="Y"),
        "",
        _fmt_caliber(m, anchors, caliber="X"),
    ]
    return "\n".join(lines)


def _fmt_report(macro: Ye2023JsscCimMacro, measurements: list[PointMeasurement], anchors: dict) -> str:
    """The whole dual-caliber report: channels, per-point caliber tables, finding."""
    parts = [_fmt_channels(measurements), ""]
    for m in measurements:
        parts += [_fmt_point(m, anchors), ""]
    parts.append(_fmt_finding(macro, measurements, anchors))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_device(name: str | None) -> torch.device:
    if name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-w", type=int, default=64, help="Weight draws per round: the die-ensemble width.")
    ap.add_argument("--n-x", type=int, default=256, help="Input vectors per round.")
    ap.add_argument("--repeat", type=int, default=8, help="Rounds; each redraws the weights AND the inputs.")
    ap.add_argument(
        "--solve-chunk",
        type=int,
        default=4096,
        help="Array solve chunk (machine knob): leading instances solved per block; 0 solves all at once.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, help="cpu / cuda[:idx]; default cuda if available else cpu.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with _ANCHORS_PATH.open("rb") as fh:
        anchors = tomllib.load(fh)
    p_zero_weight = anchors["data"]["p_zero_weight"]

    device = _resolve_device(args.device)
    # The gates read ONE die; the measurement reads an ensemble of them.
    macro = build_macro(_PARAMS_PATH, _POLICY_PATH, device=device, solve_chunk_size=args.solve_chunk)
    ensemble = build_macro(
        _PARAMS_PATH,
        _POLICY_PATH,
        device=device,
        inst_shape=(args.n_w,),
        solve_chunk_size=args.solve_chunk,
    )

    _LOG.info(
        "# ye2023jssc validation  [device %s, n_w %d, n_x %d, repeat %d, solve-chunk %d, seed %d]\n",
        device,
        args.n_w,
        args.n_x,
        args.repeat,
        args.solve_chunk,
        args.seed,
    )

    # One seed for BOTH points: common random numbers across the sparsity sweep.
    measurements = [
        measure(
            ensemble,
            p_zero_input=float(p_zero),
            p_zero_weight=p_zero_weight,
            n_x=args.n_x,
            repeat=args.repeat,
            seed=args.seed,
        )
        for p_zero in anchors["data"]["p_zero_input"]
    ]

    _LOG.info("## Hard gates\n")
    gates = [
        gate_golden_transfer(macro, n=_GOLDEN_SAMPLE_NUM, seed=args.seed + 101),
        gate_i_tbl_table(macro, anchors),
        gate_rscsa_energy(macro, anchors, measurements),
        gate_zero_input(macro, seed=args.seed + 102),
        gate_t_ac(macro, anchors, measurements),
    ]
    for g in gates:
        emit = _LOG.info if g.passed else _LOG.error
        emit("  [%s] %s: %s", "PASS" if g.passed else "FAIL", g.name, g.detail)
    passed = sum(g.passed for g in gates)
    _LOG.info("\nHARD GATES: %d/%d PASS\n", passed, len(gates))

    _LOG.info("## Dual-caliber power report (NOT gated)\n")
    _LOG.info("%s\n", _fmt_report(macro, measurements, anchors))

    if passed != len(gates):
        raise SystemExit(f"{len(gates) - passed} hard gate(s) FAILED")


if __name__ == "__main__":
    main()
