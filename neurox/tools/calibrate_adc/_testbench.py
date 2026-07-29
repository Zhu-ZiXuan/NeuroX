"""Testbench: device under test + ideal twin, stimuli, paired responses.

The testbench is registry-driven and scheme-agnostic: the tool TOML names a
macro config/policy file pair, ``CimMacroConfig.from_file`` +
``CimMacro.from_config`` resolve the concrete tile, and the two calibration
views are obtained by different means. The physical tile's analog ADC input
and code come from the
:class:`~neurox.primitive.analog.current_adc.IadcProber`; the lossless
integer dots come
straight from the RETURN VALUE of the
:meth:`~neurox.primitive.macro.cim.CimMacro.to_ideal` twin's ``vec_mat_mul``
(the ideal tile is reachable data, so it needs no side channel). Pairing
relies on the macro preserving logical-column order through exact reshapes
(the CimMacro layout contract), so the flattened per-record streams align
element for element.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from neurox.primitive.analog.current_adc import IadcProber
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy
from neurox.primitive.macro.cim.ideal import IdealCimMacro
from neurox.primitive.physical_constant import T_ROOM__K
from neurox.tools._config import resolve_relative_path

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(message)s"


def add_file_logging(log_dir: Path, tool_name: str) -> Path:
    """Attach a per-run file handler under ``log_dir`` and return its path.

    The file mirrors the console format (plain messages) so a log line can
    be pasted into a TOML unchanged.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{tool_name}_{stamp}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    return log_path


# ---------------------------------------------------------------------------
# Macro section (shared TOML schema fragment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroSection:
    """``[macro]`` section: which tile to build, by file reference.

    Attributes:
        input_num: Logical input-vector length passed to the macro constructor.
        output_num: Logical output-vector length passed to the macro constructor.
        config_files: Macro config TOML paths in descending merge priority
            (first-wins deep merge, e.g. a geometry overlay on top of the
            scheme default), relative to the tool TOML.
        config_section: Section name inside the config files holding the
            ``_neurox_class``-tagged macro config.
        policy_file: Nonideality policy TOML path (the all-off preset for
            calibration), relative to the tool TOML.
        policy_section: Section name inside ``policy_file``.
    """

    input_num: int
    output_num: int
    config_files: tuple[Path, ...]
    config_section: str
    policy_file: Path
    policy_section: str

    def __post_init__(self) -> None:
        if self.input_num < 1:
            raise ValueError(f"require: [macro].input_num ({self.input_num}) >= 1")
        if self.output_num < 1:
            raise ValueError(f"require: [macro].output_num ({self.output_num}) >= 1")
        if not self.config_files:
            raise ValueError("require: [macro].config_files non-empty")


def build_physical_macro(section: MacroSection, *, base: Path, device: torch.device) -> CimMacro:
    """Build, fabricate, and eval-freeze the physical tile named by ``section``.

    Args:
        section: The ``[macro]`` file references.
        base: The tool TOML path the relative file references resolve
            against.
        device: Target torch device.
    """
    config_paths: list[Path] = []
    for file in section.config_files:
        resolved = resolve_relative_path(file, base)
        assert resolved is not None
        config_paths.append(resolved)
    policy_path = resolve_relative_path(section.policy_file, base)
    assert policy_path is not None
    config = CimMacroConfig.from_file(*config_paths, section=section.config_section)
    policy = CimMacroPolicy.from_file(policy_path, section=section.policy_section)
    macro = CimMacro.from_config(
        config=config,
        policy=policy,
        input_num=section.input_num,
        output_num=section.output_num,
        inst_shape=(),
        # The library-wide forward dtype; calibration statistics are
        # accumulated in float64 downstream of the probe.
        dtype=torch.float32,
        T__K=T_ROOM__K,
    )
    macro = macro.to(device)
    macro.eval()
    macro.fabricate()
    logger.info(
        "built %s from %s [%s] + %s [%s]",
        type(macro).__name__,
        " <- ".join(str(p) for p in config_paths),
        section.config_section,
        policy_path,
        section.policy_section,
    )
    return macro


def build_ideal_twin(macro: CimMacro, *, device: torch.device) -> IdealCimMacro:
    """Build the lossless ideal twin (weights are programmed separately)."""
    ideal = macro.to_ideal().to(device)
    ideal.eval()
    return ideal


# ---------------------------------------------------------------------------
# Stimulus generation (CPU generator for determinism; moved by the runner)
# ---------------------------------------------------------------------------


def sample_ternary_w(gen: torch.Generator, *, col_num: int, row_num: int, density: float) -> Tensor:
    """Random ternary digit tensor ``(col_num, 1, row_num)`` at ``density``.

    Each cell is non-zero with probability ``density``; non-zero cells are
    ``+1`` or ``-1`` with equal probability.
    """
    active = torch.rand((col_num, 1, row_num), generator=gen) < density
    sign = torch.where(torch.rand((col_num, 1, row_num), generator=gen) < 0.5, -1, 1)
    return torch.where(active, sign, torch.zeros_like(sign)).to(torch.long)


def sample_capped_block_w(
    gen: torch.Generator,
    *,
    col_num: int,
    row_num: int,
    active_row_num: int,
    cap: int,
) -> Tensor:
    """Random single-sign per-(column, phase)-block ternary weights, count-capped.

    Each (column, phase) block programs ``m ~ Uniform{0 .. cap}`` cells at
    random row positions within the block, all sharing one random sign per
    block — so under full WL drive the block's per-phase MAC magnitude is
    exactly ``m`` (the single-cell-LSB battery pattern). When ``active_row_num``
    does not divide ``row_num`` the final block is short (its padded tail is
    truncated), mirroring the engine's partial last sub-phase.

    Returns:
        Digit tensor ``(col_num, 1, row_num)``.
    """
    if not (0 <= cap <= active_row_num):
        raise ValueError(f"require: 0 <= cap ({cap}) <= active_row_num ({active_row_num})")
    # Ceil so a non-divisible geometry still covers every row; the padded tail
    # is dropped after the reshape (short final block == the engine's partial
    # last sub-phase). When active_row_num divides row_num this is exact.
    phase_num = -(-row_num // active_row_num)
    counts = torch.randint(0, cap + 1, (col_num, phase_num), generator=gen)
    signs = torch.where(torch.rand((col_num, phase_num), generator=gen) < 0.5, -1, 1)
    # Rank positions per block by random score; the first `count` win.
    score = torch.rand((col_num, phase_num, active_row_num), generator=gen)
    rank = score.argsort(dim=-1).argsort(dim=-1)
    w_block = torch.where(rank < counts.unsqueeze(-1), signs.unsqueeze(-1), torch.zeros_like(signs).unsqueeze(-1))
    return w_block.reshape(col_num, 1, phase_num * active_row_num)[..., :row_num].to(torch.long)


def grid_block_w(
    *,
    col_num: int,
    row_num: int,
    active_row_num: int,
    m_max: int,
    offset: int = 0,
    col_stride: int = 1,
) -> Tensor:
    """Deterministic count-grid weights walking every ``|M|`` in ``0 .. m_max``.

    Every ``col_stride``-th column is programmed (the others stay zero — a
    loading-dilution knob keeping the total array conduction inside the
    workload envelope the DC solve converges on); programmed column ``c``'s
    (column, phase) blocks each program ``(c // col_stride + offset) %
    (m_max + 1)`` leading cells, with the sign alternating across programmed
    columns so both the P and the N polarity paths carry every magnitude.
    Under full WL drive the per-phase MAC magnitude of programmed column
    ``c`` is exactly ``(c // col_stride + offset) % (m_max + 1)``. One
    pattern covers ``min(col_num // col_stride, m_max + 1)`` distinct
    magnitudes; a caller needing full coverage runs
    ``ceil((m_max + 1) / (col_num // col_stride))`` patterns at offsets
    ``0, col_num // col_stride, ...``. When ``active_row_num`` does not divide
    ``row_num`` the final block is short (its padded tail is truncated),
    mirroring the engine's partial last sub-phase.

    Returns:
        Digit tensor ``(col_num, 1, row_num)``.
    """
    if not (1 <= m_max <= active_row_num):
        raise ValueError(f"require: 1 <= m_max ({m_max}) <= active_row_num ({active_row_num})")
    if not (1 <= col_stride <= col_num):
        raise ValueError(f"require: 1 <= col_stride ({col_stride}) <= col_num ({col_num})")
    # Ceil so a non-divisible geometry still covers every row; the padded tail
    # is dropped after the reshape (short final block == the engine's partial
    # last sub-phase). When active_row_num divides row_num this is exact.
    phase_num = -(-row_num // active_row_num)
    col = torch.arange(col_num)
    programmed = col % col_stride == 0  # (col,)
    grid_idx = col // col_stride  # (col,)
    counts = torch.where(programmed, (grid_idx + offset) % (m_max + 1), torch.zeros_like(grid_idx))  # (col,)
    signs = torch.where(grid_idx % 2 == 0, 1, -1)  # (col,)
    pos = torch.arange(active_row_num)  # (active_row,)
    w_block = torch.where(
        pos.view(1, 1, -1) < counts.view(-1, 1, 1),
        signs.view(-1, 1, 1),
        torch.zeros((), dtype=torch.long),
    ).expand(col_num, phase_num, active_row_num)
    return w_block.reshape(col_num, 1, phase_num * active_row_num)[..., :row_num].to(torch.long)


def saturating_w(*, col_num: int, row_num: int, active_row_num: int) -> Tensor:
    """Dense saturating columns: all ``+1`` / all ``-1`` / phase-antisymmetric.

    Column pattern cycles through the three saturating shapes; under full
    drive every phase saturates the readout (clips at the top code) —
    the loading-envelope extreme of the battery. When ``active_row_num`` does
    not divide ``row_num`` the final block is short (its padded tail is
    truncated), mirroring the engine's partial last sub-phase.
    """
    # Ceil so a non-divisible geometry still covers every row; the padded tail
    # is dropped after the reshape (short final block == the engine's partial
    # last sub-phase). When active_row_num divides row_num this is exact.
    phase_num = -(-row_num // active_row_num)
    kind = torch.arange(col_num) % 3
    phase_sign = torch.where(torch.arange(phase_num) % 2 == 0, 1, -1)  # (phase,)
    w_block = torch.empty((col_num, phase_num, active_row_num), dtype=torch.long)
    w_block[kind == 0] = 1
    w_block[kind == 1] = -1
    w_block[kind == 2] = phase_sign.view(1, -1, 1)
    return w_block.reshape(col_num, 1, phase_num * active_row_num)[..., :row_num]


def sample_binary_x(gen: torch.Generator, *, batch: int, row_num: int, density: float) -> Tensor:
    """Random binary WL drive plane batch ``(batch, row_num)`` at ``density``."""
    return (torch.rand((batch, row_num), generator=gen) < density).to(torch.long)


# ---------------------------------------------------------------------------
# Dual probed run
# ---------------------------------------------------------------------------


def _unroll_sub_phase(x: Tensor, *, row_num: int, max_active_num: int, inst_rank: int) -> Tensor:
    """Expand WL planes over the macro's hardware sub-phase axis.

    Local mirror of the runtime engine-layer serialization: the sub-phase
    axis ``P = ceil(row_num / max_active_num)`` is inserted immediately LEFT
    of the macro's inst-alignment span (``inst_rank`` size-1 slots), and
    rows outside a plane's active window are zeroed (WL off), so every
    conversion drives at most ``max_active_num`` live rows — the
    per-conversion drive context the ``vec_mat_mul`` contract requires.

    Args:
        x: WL plane tensor with trailing ``[row_num]``.
        row_num: Macro row count. When ``max_active_num`` does not divide it
            the final sub-phase reads the short remainder block (mirrors the
            engine's ceil sub-phase count).
        max_active_num: Maximum simultaneously active word lines per
            conversion (``CimMacro.max_active_num``).
        inst_rank: Rank of the macro's fabricated ``inst_shape``.

    Returns:
        Masked plane tensor trailing ``[P, *(1,) * inst_rank, row_num]``;
        dtype and device follow ``x``.
    """
    # Static row -> sub-phase ownership; phase p owns rows
    # [p * max_active_num, (p + 1) * max_active_num). Ceil so every real row
    # lands in exactly one sub-phase; the short final block leaves its own rows
    # the only ones live in that plane. Shape: [P, row_num]
    mask = torch.arange(row_num, device=x.device) // max_active_num == torch.arange(
        -(-row_num // max_active_num), device=x.device
    ).unsqueeze(-1)
    # Shape: [P, row_num] -> [P, *(1,) * inst_rank, row_num]
    mask = mask.reshape(-1, *(1,) * inst_rank, row_num)
    # Shape: [..., *span, row_num] -> [..., P, *span, row_num]; zero-fill = WL off
    return torch.where(mask, x.unsqueeze(max(-(inst_rank + 2), -(x.ndim + 1))), x.new_zeros(()))


@dataclass(frozen=True)
class PairedConversion:
    """Flattened, order-aligned calibration streams for one stimulus.

    Attributes:
        i_in__uA: Analog ADC input per conversion element (physical run,
            :class:`IadcProber`), CPU float64, 1-D.
        code: ADC output code per element (physical run), CPU int64, 1-D.
        ideal_m: Lossless integer per-phase dot per element (ideal run's
            ``vec_mat_mul`` return at the ``adc_bits = None`` oracle), CPU
            int64, 1-D, signed.
    """

    i_in__uA: Tensor
    code: Tensor
    ideal_m: Tensor


def run_paired_stimulus(
    physical: CimMacro,
    ideal: IdealCimMacro,
    *,
    w: Tensor,
    x: Tensor,
    input_num: int,
    quantization_mode: int,
    adc_bits: int,
) -> PairedConversion:
    """Program + run one stimulus through both tiles, pairing their views.

    Both tiles are programmed with the same digit tensor (the ideal twin
    shares no state) and driven with the same sub-phase-expanded WL
    planes (:func:`_unroll_sub_phase`, so calibration converts under the
    per-sub-phase masked drive the runtime applies and the streams stay
    element-aligned). The physical VMM runs at
    ``(quantization_mode, adc_bits)`` under a :class:`IadcProber` capturing
    the convert observations; the ideal VMM runs at the lossless
    ``adc_bits = None`` oracle and its integer-dot RETURN value is the
    ideal view (the ideal tile emits no probe). The physical observations
    and the ideal returns are paired positionally.

    Args:
        physical: Fabricated physical tile.
        ideal: Its lossless twin.
        w: Digit tensor matching the macro weight layout.
        x: Activation tensor with trailing ``[row_num]``; the testbench
            performs the sub-phase expansion internally.
        input_num: Logical input-vector length.
        quantization_mode: Quantization mode of the physical run.
        adc_bits: ADC resolution of the physical run.

    Returns:
        The flattened order-aligned streams (see :class:`PairedConversion`).

    Raises:
        ValueError: If the physical macro emitted no convert observation, if
            the physical and ideal streams disagree in count, or if a paired
            physical / ideal entry disagrees in element count (a macro that
            breaks the column-order-preserving layout contract).
    """
    device = next(physical.buffers()).device
    w = w.to(device)
    x = x.to(device)
    physical.program(w)
    ideal.program(w)
    # Runtime-parity drive: serialize each requested plane over the
    # sub-phase axis so both tiles convert at most ``max_active_num``
    # live rows per plane, exactly as the engine layer drives the macro.
    # Shape: [..., row_num] -> [..., P, *(1,) * inst_rank, row_num]
    x = _unroll_sub_phase(
        x,
        row_num=input_num,
        max_active_num=physical.max_active_num,
        inst_rank=len(physical.inst_shape),
    )
    with IadcProber() as prober, torch.no_grad():
        physical.vec_mat_mul(x, quantization_mode=quantization_mode, adc_bits=adc_bits)
        # The ideal twin is reachable data: its return is the lossless view,
        # positionally paired with the physical convert observations.
        ideal_dots: list[Tensor] = [ideal.vec_mat_mul(x, quantization_mode=quantization_mode, adc_bits=None)]

    convert_observations = prober.records
    if not convert_observations:
        raise ValueError("no paired conversions recorded — the physical macro emitted no current_adc.convert events")
    if len(convert_observations) != len(ideal_dots):
        raise ValueError(
            f"paired stream counts differ (convert {len(convert_observations)} vs ideal {len(ideal_dots)})"
        )

    i_in_parts: list[Tensor] = []
    code_parts: list[Tensor] = []
    ideal_parts: list[Tensor] = []
    for observation, ideal_dot in zip(convert_observations, ideal_dots, strict=True):
        i_in = observation.i_in__uA.flatten().to("cpu", torch.float64)
        code = observation.code.flatten().to("cpu", torch.int64)
        ideal_m = ideal_dot.flatten().to("cpu", torch.int64)
        if i_in.numel() != ideal_m.numel():
            raise ValueError(
                f"paired record element counts differ (convert {i_in.numel()} vs ideal {ideal_m.numel()}); "
                "the macro must preserve logical-column order and count through its readout reshapes"
            )
        i_in_parts.append(i_in)
        code_parts.append(code)
        ideal_parts.append(ideal_m)
    return PairedConversion(
        i_in__uA=torch.cat(i_in_parts),
        code=torch.cat(code_parts),
        ideal_m=torch.cat(ideal_parts),
    )
