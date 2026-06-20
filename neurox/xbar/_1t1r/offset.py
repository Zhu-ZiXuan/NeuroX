"""Offset-coded 1T1R crossbar tile.

See also:
    docs/reference/xbar/_1t1r/README.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcCalibrationRecord, AdcOperationPoint
from neurox.xbar.base import Xbar, XbarConfig, XbarPolicy
from neurox.xbar.readout import ReadOut, ReadOutConfig, ReadOutPolicy

from .circuit_core import CircuitCore1T1R, CircuitCore1T1RConfig, CircuitCore1T1RPolicy

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Offset1T1RXbarConfig(XbarConfig):
    """Offset-coded 1T1R xbar configuration.

    Attributes:
        w_digit_count: Digits per xbar-word.
        w_digit_radix: Positional base ``r`` of the in-tile digit combination.
        w_state_offset: Offset shift mapping ``digit -> state = digit + offset``.
        ref_group_size: Number of data columns per reference group.
        ref_location: Position of the ref column inside each group,
            in ``[0, ref_group_size]``.
        adc_calibration: Externally-calibrated ``(adc_mode, adc_bits) →
            rescale_factor`` records; ``M_ideal ≈ code · rescale_factor``
            (quantize is ``code = floor(M_ideal / rescale_factor)``).
            Lists the set of ADC operating points the tile supports.
        core_config: Owned physical-core config.
        readout_config: Owned readout-chain config.
    """

    w_digit_count: int
    w_digit_radix: int
    w_state_offset: int

    ref_group_size: int
    ref_location: int

    adc_calibration: tuple[AdcCalibrationRecord, ...]

    core_config: CircuitCore1T1RConfig
    readout_config: ReadOutConfig

    def validate(self) -> None:
        super().validate()
        self.validate_encoding()
        self.validate_ref_layout()
        self.validate_offset_vs_state_map()
        self.validate_adc_calibration()

    def validate_adc_calibration(self) -> None:
        if len(self.adc_calibration) == 0:
            raise ValueError("require: adc_calibration must contain at least one entry")
        seen: set[tuple[int, int]] = set()
        for entry in self.adc_calibration:
            key = (entry.adc_mode, entry.adc_bits)
            if key in seen:
                raise ValueError(f"adc_calibration has duplicate (adc_mode, adc_bits)={key}")
            seen.add(key)
            if not (entry.rescale_factor > 0.0):
                raise ValueError(
                    f"require: rescale_factor ({entry.rescale_factor}) > 0 for "
                    f"(adc_mode={entry.adc_mode}, adc_bits={entry.adc_bits})"
                )

    def validate_encoding(self) -> None:
        self._require_pos(self.w_digit_count, "w_digit_count")
        if not (self.w_digit_radix > 1):
            raise ValueError(f"require: w_digit_radix ({self.w_digit_radix}) > 1")
        self._require_nonneg(self.w_state_offset, "w_state_offset")

    def validate_offset_vs_state_map(self) -> None:
        """Cross-check ``w_state_offset`` against ``state_to_g_map``.

        The encoded state index is ``digit + w_state_offset``; the
        runtime digit range is ``[-offset, states - 1 - offset]``
        (see :attr:`Offset1T1RXbar.w_digit_range`). Two invariants:

        - ``w_state_offset < len(state_to_g_map)`` so the encoded
          range is non-empty.
        - ``0`` must lie inside the digit range so the reference
          column (always programmed with digit ``0``) maps to a
          legal state — equivalent to ``-offset <= 0 <=
          states - 1 - offset``, i.e. the same bound as above.
        """
        n_states = len(self.core_config.cell_config.state_to_g_map__uS)
        if not (self.w_state_offset < n_states):
            raise ValueError(
                f"require: w_state_offset ({self.w_state_offset}) < "
                f"len(state_to_g_map__uS) ({n_states}) — the reference "
                "digit 0 must map to a legal state"
            )

    def validate_ref_layout(self) -> None:
        self._require_pos(self.ref_group_size, "ref_group_size")
        if self.col_num % self.ref_group_size != 0:
            raise ValueError(f"require: col_num ({self.col_num}) divisible by ref_group_size ({self.ref_group_size})")
        if not (0 <= self.ref_location <= self.ref_group_size):
            raise ValueError(
                f"require: 0 <= ref_location ({self.ref_location}) <= ref_group_size ({self.ref_group_size})"
            )


# ---------------------------------------------------------------------------
# 2. Xbar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Offset1T1RXbarPolicy(XbarPolicy):
    """Composite policy for :class:`Offset1T1RXbar`.

    Attributes:
        core: 1T1R circuit-core nonideality policy. The chunk knob
            (``solve_chunk_size``) lives on this — see
            :class:`CircuitCore1T1RPolicy`.
        readout: Readout-chain nonideality policy.
    """

    core: CircuitCore1T1RPolicy
    readout: ReadOutPolicy


@Xbar.register_key(Offset1T1RXbarConfig)
class Offset1T1RXbar(Xbar):
    """Offset-coded 1T1R crossbar tile."""

    config: Offset1T1RXbarConfig
    logic_phys_idx: Tensor
    ref_phys_idx: Tensor

    def __init__(
        self,
        *,
        config: Offset1T1RXbarConfig,
        policy: Offset1T1RXbarPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

        prefix = self._inst_shape

        # Positional weights: ``[r⁰, r¹, ..., r^(D−1)]``.
        self.digit_weights = tuple(float(config.w_digit_radix**k) for k in range(config.w_digit_count))

        n_groups = config.col_num // config.ref_group_size
        total_logic_cols = config.col_num * config.w_digit_count
        self.n_ref_cols = n_groups
        self.physical_col_num = total_logic_cols + n_groups

        core_name = f"{name}.core"
        readout_name = f"{name}.readout"
        self.core = CircuitCore1T1R(
            config=config.core_config,
            policy=policy.core,
            name=core_name,
            w_layout_shape=(*prefix, self.physical_col_num, config.row_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.readout = ReadOut.from_config(
            config=config.readout_config,
            policy=policy.readout,
            name=readout_name,
            inst_shape=(*prefix, n_groups),
            dtype=dtype,
            T__K=T__K,
            slice_num=config.ref_group_size,
            digit_weights=self.digit_weights,
        )

        logic_phys, ref_phys = _build_ref_indices(
            n_groups=n_groups,
            group_size_logic=config.ref_group_size * config.w_digit_count,
            location_logic=config.ref_location * config.w_digit_count,
        )
        self.register_buffer("logic_phys_idx", logic_phys, persistent=False)
        self.register_buffer("ref_phys_idx", ref_phys, persistent=False)

        self._rescale_lut: dict[AdcOperationPoint, float] = {
            AdcOperationPoint(adc_mode=e.adc_mode, adc_bits=e.adc_bits): e.rescale_factor
            for e in config.adc_calibration
        }

    # -----------------------------------------------------------------
    # Value-domain semantics
    # -----------------------------------------------------------------

    @property
    def x_range(self) -> tuple[int, int]:
        """Inclusive single-cycle integer input range, derived from the WL DAC."""
        return (0, self.core.wl_dac.code_max)

    @property
    def w_digit_count(self) -> int:
        return self.config.w_digit_count

    @property
    def w_digit_radix(self) -> int:
        """Positional base ``r`` of the in-tile digit combination."""
        return self.config.w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        """Physical programmable range of one digit cell on this offset array.

        With ``S`` RRAM states and offset ``o``, ``state = digit + o``
        spans ``[0, S - 1]``, so the digit range is ``(-o, S - 1 - o)``.
        """
        offset = self.config.w_state_offset
        states = self.core.w_states
        return (-offset, states - 1 - offset)

    @property
    def adc_mode_num(self) -> int:
        """Number of supported ADC operating points; valid ``adc_mode`` values are ``[0, mode_num)``."""
        return self.readout.adc_mode_num

    @property
    def adc_max_bits(self) -> int:
        """Maximum supported ``adc_bits`` value."""
        return self.readout.adc_max_bits

    def adc_rescale_factor(self, adc_operation_point: AdcOperationPoint) -> float:
        """Rescale factor for ``adc_operation_point``; raises ``KeyError`` if uncalibrated."""
        try:
            return self._rescale_lut[adc_operation_point]
        except KeyError:
            available = sorted((op.adc_mode, op.adc_bits) for op in self._rescale_lut)
            raise KeyError(
                f"{adc_operation_point} not in adc_calibration; available (adc_mode, adc_bits): {available}"
            ) from None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def program(self, w: Tensor) -> None:
        """Write the tile's owned device buffers from one xbar-native digit tensor.

        Args:
            w: Integer digit tensor whose shape matches
                ``self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)``.
                Entries must lie in :attr:`w_digit_range`.
        """
        if tuple(w.shape) != self._w_layout_shape:
            raise ValueError(f"program() expects w.shape {self._w_layout_shape}; got {tuple(w.shape)}")
        # Data-major, digit-minor ordering along the column axis.
        # Shape: [..., col_num, w_digit_count, row_num] -> [..., col_num * w_digit_count, row_num]
        w_logic = w.flatten(-3, -2)
        w_state_idx = _insert_ref_cols(w_logic, self.logic_phys_idx, self.physical_col_num) + self.config.w_state_offset
        self.core.program(w_state_idx)

    def vec_mat_mul(self, x: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Run one VMM through the core → readout chain.

        Compile-path: yes (via macro entry). The bottleneck DC solve is
        compiled one layer down (``Solver.solve_dc``, a fixed-shape
        leaf); the data-dependent chunk loop that drives it lives in the
        eager island ``CircuitCore1T1R.cim_read``. So tracing the macro
        ``matmul`` fuses this method's index / readout math and breaks
        only at ``cim_read``. See docs/internals/compile/scheme-a-regional.md.

        Chunking is owned by the core: it reads ``solve_chunk_size`` off
        ``policy.core`` and decides whether to walk the broadcast leading
        in chunks of at most that many instances or run a single
        full-broadcast solve. The core emits exactly one energy event and
        one latency event per VMM regardless of the chunking choice.

        Args:
            x: Activation tensor with primitive trailing ``[row_num]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            ADC-code tensor with primitive trailing ``[col_num]``.
        """
        v_out_phys = self.core.cim_read(x)

        v_signal_phys = v_out_phys.index_select(-1, self.logic_phys_idx)
        v_ref_phys = v_out_phys.index_select(-1, self.ref_phys_idx)
        group_num = self.n_ref_cols
        slice_num = self.config.ref_group_size
        digit_count = self.config.w_digit_count
        v_signal_grouped = v_signal_phys.unflatten(-1, (group_num, slice_num, digit_count))

        code = self.readout.readout(
            v_signal_grouped,
            v_ref_phys,
            adc_operation_point=adc_operation_point,
        )

        return code.flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Reference-column helpers (offset-1T1R-specific layout)
# ---------------------------------------------------------------------------


def _insert_ref_cols(w: Tensor, logic_phys_idx: Tensor, physical_col_num: int) -> Tensor:
    """Scatter logic-column weight digits into the physical-width layout.

    Args:
        w: Mapped weight digits with the digit axis inlined along the
            col dimension. Shape ``[..., col_num * w_digit_count, row_num]``.
        logic_phys_idx: Long tensor mapping each logic-column position
            to its physical-column index.
        physical_col_num: Total physical-column count (logic + ref).

    Returns:
        Physical weight digits with ref columns inserted, shape
        ``[..., physical_col_num, row_num]``. Ref slots stay at zero fill.
    """
    *batch, _logic_cols, row_num = w.shape
    w_phys = torch.zeros(*batch, physical_col_num, row_num, dtype=w.dtype, device=w.device)
    w_phys.index_copy_(dim=-2, index=logic_phys_idx, source=w)
    return w_phys


def _build_ref_indices(
    n_groups: int,
    group_size_logic: int,
    location_logic: int,
) -> tuple[Tensor, Tensor]:
    """Build the physical-column index tables for the ref-column scatter.

    Each group of ``group_size_logic + 1`` physical columns holds
    ``group_size_logic`` logic columns plus one ref at ``location_logic``.

    Args:
        n_groups: Number of ref groups.
        group_size_logic: Logic columns per ref group.
        location_logic: Ref-column position within each group,
            in ``[0, group_size_logic]``.

    Returns:
        ``(logic_phys_idx, ref_phys_idx)`` Long tensors of length
        ``n_groups * group_size_logic`` and ``n_groups``.
    """
    logic_phys: list[int] = []
    ref_phys: list[int] = []
    for g in range(n_groups):
        base = g * (group_size_logic + 1)
        for p in range(group_size_logic + 1):
            if p == location_logic:
                ref_phys.append(base + p)
            else:
                logic_phys.append(base + p)
    return (
        torch.tensor(logic_phys, dtype=torch.long),
        torch.tensor(ref_phys, dtype=torch.long),
    )
