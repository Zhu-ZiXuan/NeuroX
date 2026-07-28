"""DSWCT place-value weighting stage — one digit-weighted current-mirror bank.

The bank mirrors each of its ``w_digit`` per-digit BL leg currents at the
LSB-first place-value ratio and sums the weighted legs into the output current
``I_WDL``. The ``w_digit`` legs are internal structure of one bank, not
instances, so the fabricated ``inst_shape`` trailing is ``(gn, 2)``. A reporter
leaf: it self-bills its rail conduction and cap events at the production site.

See also:
    docs/works/macro/cim/xue2020jssc/model.md
"""

from __future__ import annotations

from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase


class DswctConfig(ConfigBase):
    """Immutable configuration for :class:`Dswct`.

    Attributes:
        area_per_inst__um2: Silicon area per mirror bank.
        leakage_per_inst__uW: Static leakage per mirror bank.
        c_load__fF: Output-leg load capacitance switched once per (mux slot x
            WL bit plane) per bank; each event costs ``c_load * V_DD**2``.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float
    c_load__fF: float

    def validate(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.c_load__fF, "c_load__fF")


class DswctPolicy(PolicyBase):
    """Source-free DSWCT policy — no nonideality toggles."""


class Dswct(ModuleBase[DswctConfig, DswctPolicy]):
    """Digit-weighted current-mirror bank: place-value combine of the BL leg currents.

    Args:
        config: DSWCT configuration.
        policy: Source-free DSWCT policy.
        inst_shape: Fabrication shape ``(*inst, gn, 2)`` — one bank per
            (CIM-IO, polarity); the trailing axis is the P/N polarity pair.
        digit_ratios: LSB-first per-digit mirror ratios, 1-D tensor of length
            ``w_digit``, in the module's working dtype.
        v_dd__V: Supply-rail voltage [V] every weighted leg conducts across.
    """

    # --- Immutable model buffers ---

    _digit_ratios: Tensor

    def __init__(
        self,
        *,
        config: DswctConfig,
        policy: DswctPolicy,
        inst_shape: tuple[int, ...],
        digit_ratios: Tensor,
        v_dd__V: float,
    ) -> None:
        if len(inst_shape) < 2 or inst_shape[-1] != 2:
            raise ValueError(f"require: inst_shape ({inst_shape}) ends with (gn, 2) — one bank per (IO, polarity)")
        if digit_ratios.ndim != 1 or digit_ratios.numel() < 1:
            raise ValueError(f"require: digit_ratios is a non-empty 1-D tensor; got shape {tuple(digit_ratios.shape)}")
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self._v_dd__V = v_dd__V
        self.register_buffer("_digit_ratios", digit_ratios.detach().clone(), persistent=False)

    def _sample_fabricate_mismatch(self) -> None:
        pass

    @property
    def digit_num(self) -> int:
        """Number of place-value legs per bank — the digit-ratio buffer length."""
        return int(self._digit_ratios.shape[0])

    def forward(self, i_dl__uA: Tensor, *, window__ns: Tensor | float) -> Tensor:
        """Weight the per-digit BL currents by place value and sum over digits.

        The leading batch carries the WL bit-plane axis (and any data batch), so
        each leading element is one plane's conduction.

        Args:
            i_dl__uA: Per-lane BL port current [uA], shape
                ``[..., serial, gn, 2, w_digit]``; every leading axis is
                anonymous broadcast batch.
            window__ns: Conduction window [ns] of this call's plane(s);
                broadcasts against the leading batch shape ``[...]`` left after
                the trailing ``(serial, gn, 2, w_digit)`` reduction.

        Returns:
            Digit-combined output current ``I_WDL`` [uA], shape
            ``[..., serial, gn, 2]``.
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
            # Rail: V_DD * |I_WDL| * window over every (slot, lane, digit) leg.
            # Shape: [..., serial, gn, 2, w_digit] -> [...]
            i_wdl_total__uA = i_wdl__uA.abs().sum(dim=(-4, -3, -2, -1))
            e__fJ = self._v_dd__V * i_wdl_total__uA * window__ns
            # Cap: c_load * V_DD**2 once per (slot x plane) per bank; each
            # leading element is one plane, its trailing holds slot x bank.
            serial_num, gn, pol = i_dl__uA.shape[-4:-1]
            e__fJ = e__fJ + self.config.c_load__fF * self._v_dd__V**2 * (serial_num * gn * pol)
            self._record_dynamic_energy(e__fJ)
        # Shape: [..., serial, gn, 2, w_digit] -> [..., serial, gn, 2]
        return i_wdl__uA.sum(dim=-1)
