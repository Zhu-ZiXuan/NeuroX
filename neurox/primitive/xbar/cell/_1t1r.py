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

# ---------------------------------------------------------------------------
# Config / policy / result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
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
        self.validate_node_caps()

    def validate_node_caps(self) -> None:
        self._require_non_neg(self.c_bl__fF, "c_bl__fF")
        self._require_non_neg(self.c_x__fF, "c_x__fF")
        self._require_non_neg(self.c_sl__fF, "c_sl__fF")
        self._require_non_neg(self.c_wl__fF, "c_wl__fF")


@dataclass(frozen=True)
class XbarCell1t1rPolicy(XbarCellPolicy, ABC):
    """Abstract marker base for 1T1R cell nonideality policies.

    Each concrete 1T1R cell model carries its own subclass bundling the
    per-device policies of the devices it owns (possibly none).
    """


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


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


class XbarCell1t1r(
    XbarCell[ConfigT, PolicyT, CellSnapT, XbarCell1t1rDcop],
    RegistryMixin[type[XbarCell1t1rConfig], "XbarCell1t1r"],
    Generic[ConfigT, PolicyT, CellSnapT],
    ABC,
):
    """Abstract series access-device + storage 1T1R cell with a condensed branch.

    Owns the shared substrate of every 1T1R model: the four per-cell
    node-to-ground capacitances and the grounded-cap switching-energy
    formula over all four cell nodes (BL, internal X, SL, and the WL
    NMOS gate the cell owns); the WL wire charge is billed by the
    owning array. Concrete leaves supply the
    branch physics (``snapshot`` /
    ``program`` / ``solve_branch`` / ``solve_dc``) and must derive and
    set ``w_states`` in ``__init__``.
    """

    w_states: int
    """Programmable weight-state count; each leaf derives and sets it in ``__init__``."""

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

        self.c_bl__fF = config.c_bl__fF
        self.c_x__fF = config.c_x__fF
        self.c_sl__fF = config.c_sl__fF
        self.c_wl__fF = config.c_wl__fF

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

        Family-bounded dispatch: the key domain is the 1T1R config
        subtree, so the resolved impl is always an :class:`XbarCell1t1r`
        leaf — the owning array needs no post-build type narrowing.
        """
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    # -----------------------------------------------------------------
    # Dynamic energy
    # -----------------------------------------------------------------

    def dynamic_energy(
        self,
        v_bl: Tensor,
        v_sl: Tensor,
        dcop: XbarCell1t1rDcop,
        snap: XbarCell1t1rSnap,
    ) -> Tensor:
        """Per-cell node-capacitance switching energy [fJ].

        Sums the grounded ``C·V²`` switching terms of the cell's nodes
        (BL, internal X, SL, WL gate) at the converged operating point
        (``V_BL``, ``V_SL``, the condensed ``V_X``, and the WL drive),
        assuming a full 0 → DC → 0 charge/discharge cycle per node.
        Pure computation: the owning array is the sole logger of every
        cell energy term — the cell emits nothing itself.

        Args:
            v_bl: Bit-line node voltage [V]. Shape: ``[..., col, row]``.
            v_sl: Source-line node voltage [V]. Shape: ``[..., col, row]``.
            dcop: Converged DCOP carrying ``v_x__V``.
            snap: Per-call snap from :meth:`snapshot`.

        Returns:
            Per-cell switching energy [fJ]. Shape: ``[..., col, row]``.
        """
        e_bl__fJ = self.c_bl__fF * v_bl.square()
        e_x__fJ = self.c_x__fF * dcop.v_x__V.square()
        e_sl__fJ = self.c_sl__fF * v_sl.square()
        e_wl__fJ = self.c_wl__fF * snap.v_wl__V.square()

        return e_bl__fJ + e_x__fJ + e_sl__fJ + e_wl__fJ
