"""Offset-coded 1T1R crossbar tile.

See also:
    docs/dev/modules/xbar/_1t1r/README.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.readout import ReadOut, ReadOutConfig
from neurox.xbar.base import Xbar, XbarConfig
from neurox.xbar.ideal import IdealXbar

from .circuit_core import CircuitCore1T1R, CircuitCore1T1RConfig, Core1T1RDCOP

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
        if not (self.adc_bits >= 1):
            raise ValueError(f"require: adc_bits ({self.adc_bits}) >= 1")

    def validate_ref_layout(self) -> None:
        self._require_pos(self.ref_group_size, "ref_group_size")
        if self.col_num % self.ref_group_size != 0:
            raise ValueError(
                f"require: col_num ({self.col_num}) divisible by ref_group_size ({self.ref_group_size})"
            )
        if not (0 <= self.ref_location <= self.ref_group_size):
            raise ValueError(
                f"require: 0 <= ref_location ({self.ref_location}) <= ref_group_size ({self.ref_group_size})"
            )


# ---------------------------------------------------------------------------
# 2. Xbar
# ---------------------------------------------------------------------------


class Offset1T1RXbar(Xbar):
    """Offset-coded 1T1R crossbar tile."""

    logic_phys_idx: Tensor
    ref_phys_idx: Tensor
    digit_weights: Tensor

    def __init__(
        self,
        *,
        cfg: Offset1T1RXbarConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        stochastic: bool | None,
    ) -> None:
        super().__init__(cfg, name=name)

        self.cfg = cfg
        self.dtype = dtype
        self.T__K = T__K
        self.stochastic: bool | None = stochastic

        # Build the two owned children from the embedded member configs.
        core_name = f"{name}.core"
        readout_name = f"{name}.readout"
        self.core: CircuitCore1T1R = CircuitCore1T1R(
            cfg=cfg.core_cfg,
            name=core_name,
            T__K=T__K,
            dtype=dtype,
        )
        self.readout: ReadOut = ReadOut.from_config(
            cfg=cfg.readout_cfg,
            name=readout_name,
            T__K=T__K,
            dtype=dtype,
            stochastic=stochastic,
        )

        # --- Reference-column index tables ---
        # Layout per group of ``ref_group_size`` data:
        #   [d_0.0 ... d_0.{D-1}, d_1.0 ... d_1.{D-1}, ...]
        # with one ref column at the data-index position
        # ``ref_location`` of each group.
        n_groups = cfg.col_num // cfg.ref_group_size
        total_logic_cols = cfg.col_num * cfg.w_digit_count
        self.n_ref_cols: int = n_groups
        self.physical_col_num: int = total_logic_cols + n_groups

        logic_phys, ref_phys = _build_ref_indices(
            n_groups=n_groups,
            group_size_logic=cfg.ref_group_size * cfg.w_digit_count,
            location_logic=cfg.ref_location * cfg.w_digit_count,
        )
        self.register_buffer("logic_phys_idx", logic_phys, persistent=False)
        self.register_buffer("ref_phys_idx", ref_phys, persistent=False)

        # Per-digit weight vector ``[radix^0, radix^1, ..., radix^(D-1)]``
        # of the offset code.  Registered as a non-persistent buffer at
        # init so ``module.to(device)`` migrates it with the rest of
        # the xbar.
        digit_weights = torch.tensor(
            [cfg.w_digit_radix**k for k in range(cfg.w_digit_count)],
            dtype=torch.int32,
        )
        self.register_buffer("digit_weights", digit_weights, persistent=False)

    # -----------------------------------------------------------------
    # Value-domain semantics
    # -----------------------------------------------------------------

    @property
    def x_range(self) -> tuple[int, int]:
        """1T1R activation grid — binary WL pulses per VMM cycle."""
        return (0, 1)

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

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def to_ideal(self) -> IdealXbar:
        """Build the noise-free :class:`IdealXbar` twin of this tile."""
        return IdealXbar(
            cfg=XbarConfig(
                col_num=self.cfg.col_num,
                row_num=self.cfg.row_num,
                adc_mode=self.cfg.adc_mode,
                adc_bits=self.cfg.adc_bits,
                output_rescale_factors=self.cfg.output_rescale_factors,
            ),
            name=self.qualified_name,
            x_range=self.x_range,
            w_digit_count=self.w_digit_count,
            w_digit_radix=self.w_digit_radix,
            w_digit_range=self.w_digit_range,
        )

    def fabricate(self, w: Tensor) -> None:
        """Lay out an xbar-native digit tensor onto the physical array.

        Args:
            w: Xbar-native digit tensor in :attr:`w_digit_range`,
                shape ``[..., col_num, w_digit_count, row_num]``.
        """
        # Record the physical xbar instance count for static aggregation.
        # ``prod(w.shape[:-3])`` covers every leading dim (macro batch /
        # M / Tc / Tr / Sa / Sw) per the macro's tile-shape contract.
        self._record_xbar_inst_count(w)
        # Data-major, digit-minor ordering along the column axis.
        # Shape: [..., col_num, w_digit_count, row_num] -> [..., col_num * w_digit_count, row_num]
        w_logic = w.flatten(-3, -2)
        w_phys = _insert_ref_cols(w_logic, self.logic_phys_idx, self.physical_col_num) + self.cfg.w_state_offset
        self.core.fabricate(w_phys)

        prefix = tuple(w_phys.shape[:-2])
        group_num = self.n_ref_cols
        data_num = self.cfg.ref_group_size
        self.readout.fabricate(
            (*prefix, group_num),
            data_num=data_num,
            digit_weights=self.digit_weights,
        )

    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Run one VMM through the core → readout chain.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.

        Returns:
            ADC-code tensor with primitive trailing ``[data_num]``.
        """
        # Shape: [..., row_num] -> [..., 1, row_num].
        core_dcop: Core1T1RDCOP = self.core.solve_dc(x.unsqueeze(-2))

        v_data_phys = core_dcop.v_out_phys.index_select(-1, self.logic_phys_idx)
        v_ref_phys = core_dcop.v_out_phys.index_select(-1, self.ref_phys_idx)
        group_num = self.n_ref_cols
        data_num = self.cfg.ref_group_size
        digit_num = self.cfg.w_digit_count
        v_data_grouped = v_data_phys.unflatten(-1, (group_num, data_num, digit_num))

        code = self.readout.readout(
            v_data_grouped,
            v_ref_phys,
            adc_mode=self._adc_mode,
            adc_bits=self._adc_bits,
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
