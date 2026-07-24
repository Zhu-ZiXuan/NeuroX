"""Abstract 1T1R crossbar cell — shared config, result types, and energy model.

See also:
    docs/reference/primitive/xbar/cell/_1t1r/cell.md
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin

from .base import (
    XbarCell,
    XbarCellConfig,
    XbarCellDcop,
    XbarCellPolicy,
    XbarCellSnap,
)


class XbarCell1t1rConfig(XbarCellConfig, ABC):
    """Node-to-ground capacitance knobs shared by every 1T1R cell model.

    Node-centric per-cell totals: each field is the total capacitance to
    ground seen at one of the cell's four nodes.

    Attributes:
        c_bl__fF: Per-cell node-to-ground total capacitance at the BL
            node.
        c_x__fF: Per-cell node-to-ground total capacitance at the
            internal access node X.
        c_sl__fF: Per-cell node-to-ground total capacitance at the SL
            node.
        c_wl__fF: Per-cell node-to-ground total capacitance at the WL
            node (NMOS gate load).
    """

    c_bl__fF: float
    c_x__fF: float
    c_sl__fF: float
    c_wl__fF: float

    def validate(self) -> None:
        self._require_non_neg(self.c_bl__fF, "c_bl__fF")
        self._require_non_neg(self.c_x__fF, "c_x__fF")
        self._require_non_neg(self.c_sl__fF, "c_sl__fF")
        self._require_non_neg(self.c_wl__fF, "c_wl__fF")


class XbarCell1t1rPolicy(XbarCellPolicy, ABC):
    """Base policy for 1T1R cell nonidealities."""


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rSnap(XbarCellSnap):
    """Per-call snap base of a 1T1R cell's fabricated state.

    Attributes:
        v_wl__V: Word-line drive voltage at the NMOS gate. Broadcasts
            to ``[..., col, row]``.
    """

    v_wl__V: Tensor


@dataclass(frozen=True)
class XbarCell1t1rDcop(XbarCellDcop):
    """1T1R branch working point with the condensed access-node voltage.

    Attributes:
        v_x__V: Access-node voltage (NMOS drain / RRAM bottom).
            Shape: ``[..., col, row]``.
    """

    v_x__V: Tensor


CellSnapT = TypeVar("CellSnapT", bound=XbarCell1t1rSnap)
ConfigT = TypeVar("ConfigT", bound=XbarCell1t1rConfig)
PolicyT = TypeVar("PolicyT", bound=XbarCell1t1rPolicy)


class XbarCell1t1r(
    XbarCell[ConfigT, PolicyT, CellSnapT, XbarCell1t1rDcop],
    RegistryMixin[type[XbarCell1t1rConfig], "XbarCell1t1r"],
    Generic[ConfigT, PolicyT, CellSnapT],
    ABC,
):
    """Base class for condensed series access-device and storage cells.

    Args:
        config: Concrete 1T1R cell configuration.
        policy: Composite per-device nonideality policy.
        inst_shape: Per-instance shape ``(*prefix, col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.

    Attributes:
        w_state_num: Number of programmable weight states.
    """

    # --- Subclass contract ---

    w_state_num: int

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

    @classmethod
    def from_config(
        cls,
        *,
        config: XbarCell1t1rConfig,
        policy: XbarCell1t1rPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> XbarCell1t1r:
        """Build the 1T1R cell registered for ``type(config)``.

        Args:
            config: Concrete 1T1R cell configuration.
            policy: Composite per-device nonideality policy.
            inst_shape: Per-instance shape ``(*prefix, col, row)``.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered 1T1R cell implementation.
        """
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    def compute_dynamic_energy(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        dcop: XbarCell1t1rDcop,
        snap: XbarCell1t1rSnap,
    ) -> Tensor:
        """Per-cell node-capacitance switching energy [fJ].

        Uses grounded ``C·V²`` terms for BL, X, SL, and WL under a full
        0 → DC → 0 cycle.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]. Shape: ``[..., col, row]``.
            dcop: Converged DCOP carrying ``v_x__V``.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            Per-cell switching energy [fJ]. Shape: ``[..., col, row]``.
        """
        config = self.config
        e_bl__fJ = config.c_bl__fF * v_bl.square()
        e_x__fJ = config.c_x__fF * dcop.v_x__V.square()
        e_sl__fJ = config.c_sl__fF * v_sl.square()
        e_wl__fJ = config.c_wl__fF * snap.v_wl__V.square()

        return e_bl__fJ + e_x__fJ + e_sl__fJ + e_wl__fJ
