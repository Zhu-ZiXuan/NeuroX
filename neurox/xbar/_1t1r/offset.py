"""Offset-coded 1T1R crossbar tile.

See also:
    docs/dev/modules/xbar/_1t1r/README.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.adc import AdcOperationPoint
from neurox.analog.readout import ReadOut, ReadOutConfig
from neurox.xbar.base import Xbar, XbarConfig

from .circuit_core import CircuitCore1T1R, CircuitCore1T1RConfig

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
        core_cfg: Owned physical-core config.
        readout_cfg: Owned readout-chain config.
    """

    w_digit_count: int
    w_digit_radix: int
    w_state_offset: int

    ref_group_size: int
    ref_location: int

    core_cfg: CircuitCore1T1RConfig
    readout_cfg: ReadOutConfig

    def validate(self) -> None:
        super().validate()
        self.validate_encoding()
        self.validate_ref_layout()

    def validate_encoding(self) -> None:
        self._require_pos(self.w_digit_count, "w_digit_count")
        if not (self.w_digit_radix > 1):
            raise ValueError(f"require: w_digit_radix ({self.w_digit_radix}) > 1")
        self._require_nonneg(self.w_state_offset, "w_state_offset")

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


@Xbar.register_key(Offset1T1RXbarConfig)
class Offset1T1RXbar(Xbar):
    """Offset-coded 1T1R crossbar tile."""

    cfg: Offset1T1RXbarConfig
    logic_phys_idx: Tensor
    ref_phys_idx: Tensor

    def __init__(
        self,
        *,
        cfg: Offset1T1RXbarConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(cfg=cfg, name=name, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

        prefix = self._inst_shape

        # Positional weights: ``[r^0, r^1, ..., r^(D-1)]``.
        self.digit_weights = tuple(float(cfg.w_digit_radix**k) for k in range(cfg.w_digit_count))

        n_groups = cfg.col_num // cfg.ref_group_size
        total_logic_cols = cfg.col_num * cfg.w_digit_count
        self.n_ref_cols = n_groups
        self.physical_col_num = total_logic_cols + n_groups

        core_name = f"{name}.core"
        readout_name = f"{name}.readout"
        self.core = CircuitCore1T1R(
            cfg=cfg.core_cfg,
            name=core_name,
            w_layout_shape=(*prefix, self.physical_col_num, cfg.row_num),
            dtype=dtype,
            T__K=T__K,
        )
        self.readout = ReadOut.from_config(
            cfg=cfg.readout_cfg,
            name=readout_name,
            inst_shape=(*prefix, n_groups),
            dtype=dtype,
            T__K=T__K,
            data_num=cfg.ref_group_size,
            digit_weights=self.digit_weights,
        )

        logic_phys, ref_phys = _build_ref_indices(
            n_groups=n_groups,
            group_size_logic=cfg.ref_group_size * cfg.w_digit_count,
            location_logic=cfg.ref_location * cfg.w_digit_count,
        )
        self.register_buffer("logic_phys_idx", logic_phys, persistent=False)
        self.register_buffer("ref_phys_idx", ref_phys, persistent=False)

        self._log_static()

    # -----------------------------------------------------------------
    # Value-domain semantics
    # -----------------------------------------------------------------

    @property
    def x_range(self) -> tuple[int, int]:
        """Inclusive single-cycle integer input range, derived from the WL DAC."""
        return (0, self.core.wl_dac.code_max)

    @property
    def w_digit_count(self) -> int:
        return self.cfg.w_digit_count

    @property
    def w_digit_radix(self) -> int:
        """Positional base ``r`` of the in-tile digit combination."""
        return self.cfg.w_digit_radix

    @property
    def w_digit_range(self) -> tuple[int, int]:
        """Physical programmable range of one digit cell on this offset array.

        With ``S`` RRAM states and offset ``o``, ``state = digit + o``
        spans ``[0, S - 1]``, so the digit range is ``(-o, S - 1 - o)``.
        """
        offset = self.cfg.w_state_offset
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

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def program(self, w: Tensor) -> None:
        """Lay out an xbar-native digit tensor onto the physical array.

        Args:
            w: Xbar-native digit tensor in :attr:`w_digit_range`, shape
                matching ``self._w_layout_shape =
                (*inst_shape, col_num, w_digit_count, row_num)``.
        """
        if tuple(w.shape) != self._w_layout_shape:
            raise ValueError(f"program() expects w.shape {self._w_layout_shape}; got {tuple(w.shape)}")
        # Data-major, digit-minor ordering along the column axis.
        # Shape: [..., col_num, w_digit_count, row_num] -> [..., col_num * w_digit_count, row_num]
        w_logic = w.flatten(-3, -2)
        w_state_idx = _insert_ref_cols(w_logic, self.logic_phys_idx, self.physical_col_num) + self.cfg.w_state_offset
        self.core.program(w_state_idx)

    def vec_mat_mul(self, x: Tensor, *, adc_operation_point: AdcOperationPoint) -> Tensor:
        """Run one VMM through the core → readout chain.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.
            adc_operation_point: Runtime ADC operating point.

        Returns:
            ADC-code tensor with primitive trailing ``[col_num]``.
        """
        core_dcop = self.core.solve_dc(x)

        v_data_phys = core_dcop.v_out_phys.index_select(-1, self.logic_phys_idx)
        v_ref_phys = core_dcop.v_out_phys.index_select(-1, self.ref_phys_idx)
        group_num = self.n_ref_cols
        data_num = self.cfg.ref_group_size
        digit_num = self.cfg.w_digit_count
        v_data_grouped = v_data_phys.unflatten(-1, (group_num, data_num, digit_num))

        code = self.readout.readout(
            v_data_grouped,
            v_ref_phys,
            adc_operation_point=adc_operation_point,
        )

        y = code.flatten(start_dim=-2)
        return y


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


def _split_logic_and_ref(
    x: Tensor,
    logic_phys_idx: Tensor,
    ref_phys_idx: Tensor,
) -> tuple[Tensor, Tensor]:
    """Split a per-physical-column tensor into logic + ref slots.

    Args:
        x: Per-physical-column tensor, shape ``[*batch, physical_col_num]``.
        logic_phys_idx: Logic-column → physical-index map,
            shape ``[col_num * w_digit_count]``.
        ref_phys_idx: Ref-group → physical-index map, shape ``[n_groups]``.

    Returns:
        ``(x_logic, x_ref)``, shapes
        ``[*batch, col_num * w_digit_count]`` and ``[*batch, n_groups]``.
    """
    x_logic = x.index_select(-1, logic_phys_idx)
    x_ref = x.index_select(-1, ref_phys_idx)
    return x_logic, x_ref


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
