"""DSWCT place-value weighting stage."""

from __future__ import annotations

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class DswctConfig(ConfigBase):
    pass


class DswctPolicy(PolicyBase):
    pass


class Dswct(ModuleBase[DswctConfig, DswctPolicy]):
    """Digit-weighted current-mirror bank: place-value combine of the BL leg currents.

    Args:
        inst_shape: Fabrication shape `(*inst_shape, gn, polarity)` — one bank per
            (CIM-IO, polarity).
        digit_ratios: LSB-first per-digit mirror ratios, in the module's working dtype.
            Shape: `[w_digit]`.
        vdd__V: Supply rail every weighted leg conducts across.
    """

    # === Functional buffers ===

    _digit_ratios: Tensor  # Shape: [w_digit]

    def __init__(
        self,
        *,
        config: DswctConfig,
        policy: DswctPolicy,
        inst_shape: tuple[int, ...],
        digit_ratios: Tensor,
        vdd__V: float,
    ) -> None:
        if len(inst_shape) < 2 or inst_shape[-1] != 2:
            raise ValueError(f"require: inst_shape ({inst_shape}) ends with (gn, 2) — one bank per (IO, polarity)")
        if digit_ratios.ndim != 1 or digit_ratios.numel() < 1:
            raise ValueError(f"require: digit_ratios is a non-empty 1-D tensor; got shape {tuple(digit_ratios.shape)}")
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._vdd__V = vdd__V
        self._register_nonpersistent_buffer("_digit_ratios", digit_ratios.detach().clone())

    @property
    def _area_per_inst__um2(self) -> float:
        return 0.0

    @property
    def _leakage_per_inst__uW(self) -> float:
        return 0.0

    @property
    def digit_num(self) -> int:
        """Number of place-value legs per bank — the digit-ratio buffer length."""
        return int(self._digit_ratios.shape[0])

    def forward(self, i_dl__uA: Tensor, *, window__ns: tuple[float, ...] | float) -> Tensor:
        """Weight the per-digit BL currents by place value.

        The leading batch carries the WL bit-plane axis (and any data batch), so
        each leading element is one plane's conduction.

        Args:
            i_dl__uA: Per-lane BL port current; every leading axis is anonymous
                broadcast batch.
                Shape: `[..., serial, gn, polarity, w_digit]`.
            window__ns: One conduction window shared by every leading position,
                or one scalar per position of the rightmost leading plane axis.

        Returns:
            Per-digit weighted current `I_WDL`.
            Shape: `[..., serial, gn, polarity, w_digit]`.
        """
        if i_dl__uA.ndim < 4:
            raise ValueError(f"forward() expects [..., serial, gn, 2, w_digit]; got shape {tuple(i_dl__uA.shape)}")
        lane_shape = tuple(i_dl__uA.shape[-3:-1])
        if lane_shape != self.inst_shape[-2:] or i_dl__uA.shape[-1] != self.digit_num:
            raise ValueError(
                f"forward() expects trailing (gn, 2, w_digit) = "
                f"{(*self.inst_shape[-2:], self.digit_num)}; got shape {tuple(i_dl__uA.shape)}"
            )

        # LSB-first per-digit mirror ratios.
        i_wdl__uA = i_dl__uA * self._digit_ratios
        if self._is_dynamic_energy_profile_active():
            # Rail: VDD * |I_WDL| * window over every (slot, lane, digit) leg.
            # Shape: [..., serial, gn, polarity, w_digit] -> [..., serial, gn, polarity]
            i_wdl_bank__uA = i_wdl__uA.abs().sum(dim=-1)
            if isinstance(window__ns, tuple):
                plane_num = len(window__ns)
                if i_wdl_bank__uA.ndim < 4:
                    raise ValueError("forward() expects a rightmost leading plane axis for per-plane windows")
                plane_extent = i_wdl_bank__uA.shape[-4]
                if plane_extent != plane_num:
                    raise ValueError(
                        f"forward() expects the rightmost leading plane extent ({plane_extent}) "
                        f"to match the window count ({plane_num})"
                    )
                # Fold the internal plane axis with its Python-scalar windows;
                # torch.compile unrolls this fixed-length loop.
                e__fJ = i_wdl_bank__uA.select(-4, 0) * window__ns[0]
                for plane, plane_window__ns in enumerate(window__ns[1:], start=1):
                    e__fJ = e__fJ + i_wdl_bank__uA.select(-4, plane) * plane_window__ns
            else:
                e__fJ = i_wdl_bank__uA * window__ns
            e__fJ = self._vdd__V * e__fJ
            self._record_dynamic_energy(e__fJ)
        return i_wdl__uA
