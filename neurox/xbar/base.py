"""Abstract physical crossbar (xbar) interface for CiM array simulation.

The :class:`Xbar` is the digital-analog-digital VMM primitive.  It
accepts tensors whose **primitive trailing dims** already match the
tile's input grid and produces an ADC-coded output along the data
direction.  Every macro-level concern — activation serializing,
weight slicing, signed-digit transcoding, tile broadcast — lives
above this primitive in the macro / mapper layer.

Row / column terminology (topology-agnostic)
--------------------------------------------
The generic xbar defines its row and column axes by *function*, not
by any specific 1T1R / 1T2R / differential wiring:

* a **row** is the set of cells that share the same input ``x``;
* a **column** is the set of cells whose contributions aggregate
  into one output.

For the bundled 1T1R family these happen to land on the WL-shared
input lines and the BL-aggregated output lines respectively, but
that mapping lives in 1T1R-specific modules — not in this generic
contract.

Primitive shape contract (the xbar's externally-visible surface)
----------------------------------------------------------------
* :meth:`fabricate(w)` — weight digit tensor with primitive trailing
  ``[data_num, digit_num, row_num]``.  Entries must lie in
  :attr:`w_digit_range`.
* :meth:`vec_mat_mul(x)` — activation tensor with primitive trailing
  ``[row_num]``; returns an output tensor with primitive trailing
  ``[data_num]``.  Entries of ``x`` must lie in :attr:`x_range`.

These are the **only** shape semantics the generic xbar publishes.
Macro-side axes (``Bw``, ``Bx``, ``M``, ``Tc``, ``Tr``, ``Sa``,
``Sw``) belong to the macro's broadcast contract — see
:mod:`neurox.macro.xbar_macro` and :mod:`neurox.mapper.xbar` — and
are broadcast naturally by concrete xbar implementations against
the fabricated per-cell state.

Value-domain capabilities every xbar exposes
--------------------------------------------
* :attr:`x_range` — single-cycle integer input grid the tile can
  carry.  Encoding-independent.
* :attr:`w_digit_count` — number of digits per xbar-word.
* :attr:`w_digit_radix` — positional base ``r`` of the in-tile
  digit combination.
* :attr:`w_digit_range` — inclusive integer range a single digit
  cell can carry physically (set by the array structure and the
  device's state count, not by any signed-digit encoding choice).

The xbar does **not** expose an aggregate "full logical ``w``
range".  That range is an encoding-dependent quantity — it depends
on the macro / mapper's choice of transcoder, slicing, and signed-
digit policy — and is therefore the mapper's responsibility, not
the xbar's.

Fabricate contract
------------------
:meth:`Xbar.fabricate(w)` accepts a digit tensor whose primitive
trailing dims are ``[data_num, digit_num, row_num]``.  The macro /
mapper is responsible for producing this tensor from a high-
precision logical weight via its own slicing + transcoding pipeline.
The xbar then performs the bookkeeping the analog tile actually
owns — physical-column layout, reference-column insertion, RRAM
state-index offset, and forwarding the laid-out tensor to its
owned submodules.

Output rescale lookup
---------------------
The mapping from the tile's analog output to the ADC's integer
code is an externally-calibrated property.  Each
:class:`XbarConfig` carries an
:attr:`XbarConfig.output_rescale_factors` table keyed by
``(adc_mode, adc_bits)``; :attr:`Xbar.output_rescale_factor`
returns the entry matching the configured runtime operating point.
The operator's integer-quantisation pipeline folds this factor
into its multiplier / shift / bias parameters at HAT-forward /
checkpoint time.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch.nn as nn
from torch import Tensor

from neurox.profiler import ProfiledModule


@dataclass(frozen=True)
class XbarRescaleEntry:
    """One row of the ``(adc_mode, adc_bits) -> rf`` lookup table.

    Attributes:
        adc_mode: ADC operating-point index (e.g. multi-V_ref SAR
            reference selection).
        adc_bits: Active ADC bit width.
        rf: Rescale factor such that ``floor(M_ideal * rf) == code``
            where ``M_ideal`` is the ideal tile-level output value.
    """

    adc_mode: int
    adc_bits: int
    rf: float


@dataclass(frozen=True)
class XbarConfig:
    """Geometry, runtime operating point and PPA for one xbar tile.

    Row / column terminology follows the generic xbar contract — see
    :class:`Xbar`'s module docstring.

    Attributes:
        col_num: Number of columns per tile (cells aggregating to
            one output).
        row_num: Number of rows per tile (cells sharing one input).
        adc_mode: Runtime ADC operating-point index; also selects
            the active entry in :attr:`output_rescale_factors`.
        adc_bits: Runtime ADC bit width; second key for the rescale
            lookup.  Defaults to ``0`` (no ADC, e.g. ideal tiles).
        output_rescale_factors: Externally-calibrated ``(adc_mode,
            adc_bits) -> rf`` entries.  ``rf`` satisfies
            ``floor(M_ideal * rf) == code``.  The lookup is strict:
            when the active ``(adc_mode, adc_bits)`` row is missing,
            :attr:`Xbar.output_rescale_factor` raises ``KeyError`` —
            silently defaulting to ``1.0`` would corrupt the
            operator's integer quantisation pipeline.  Missing data
            here means the tile has not been calibrated for the
            requested operating point, which is a configuration
            error rather than a fallback condition.
        latency_per_op__ns: Array read latency per op [ns].
        leakage_per_inst__uW: Static leakage per tile instance [uW].
        area_per_inst__um2: Silicon area per tile instance [um2].
    """

    col_num: int
    row_num: int

    adc_mode: int = 0
    adc_bits: int = 0
    output_rescale_factors: tuple[XbarRescaleEntry, ...] = field(default_factory=tuple)

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0

    def __post_init__(self) -> None:
        # The 1T1R wire-Newton solver requires at least two nodes on
        # each wire (the IR-drop computation uses ``torch.diff`` along
        # both axes, and the analytic structure of the resistor ladder
        # assumes a driver-segment plus at least one inter-node
        # segment).  Single-row / single-column tiles are not physically
        # interesting for the simulator either, so we forbid them
        # uniformly at the base-config layer.
        if not (self.col_num > 1):
            raise ValueError(f"require: col_num ({self.col_num}) > 1")
        if not (self.row_num > 1):
            raise ValueError(f"require: row_num ({self.row_num}) > 1")
        if self.adc_mode < 0:
            raise ValueError(f"require: adc_mode ({self.adc_mode}) >= 0")
        if self.adc_bits < 0:
            raise ValueError(f"require: adc_bits ({self.adc_bits}) >= 0")


class Xbar(nn.Module, ProfiledModule, ABC):
    """Abstract base class for a physical crossbar tile.

    Concrete subclasses expose the four value-domain capability
    properties — :attr:`x_range`, :attr:`w_digit_count`,
    :attr:`w_digit_radix`, :attr:`w_digit_range` — plus the two
    geometry capability properties :attr:`col_num` and
    :attr:`row_num` (tile output / input side dimensions; size
    the array, not the value grid), and implement the
    :meth:`fabricate` / :meth:`vec_mat_mul` lifecycle.  The
    base intentionally exposes no encoding details (state counts,
    reference-column scheme, RRAM state offsets); those are the
    subclass's private property.

    The tile owns no per-cell tensor state of its own — every
    fabricated quantity (RRAM conductance, NMOS β / V_th, ideal-tile
    weights) lives on the device-level submodule whose physics it
    represents.  The xbar wires those submodules together and exposes
    the fabricate / forward contract.

    Args:
        cfg: Tile geometry, runtime ADC operating point, and PPA.
        name: Hierarchical instance name used by the profiler.  Child
            modules built inside ``__init__`` should derive their own
            names from this one.
    """

    def __init__(self, cfg: XbarConfig, *, name: str = "") -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        self.config = cfg

        self.col_num = cfg.col_num
        self.row_num = cfg.row_num
        # Runtime ADC operating point — drives ``bl_adc.convert /
        # latency_per_op__ns`` for physical tiles and the rescale
        # lookup below for every tile.
        self._adc_mode: int = cfg.adc_mode
        self._adc_bits: int = cfg.adc_bits
        # Dict view of the lookup tuple, built once for cheap runtime
        # access.  Empty when the config carries no entries.
        self._rescale_lut: dict[tuple[int, int], float] = {
            (e.adc_mode, e.adc_bits): e.rf for e in cfg.output_rescale_factors
        }

    # ----- PPA properties (delegated to the immutable config) -----

    @property
    def area_per_inst__um2(self) -> float:
        """Area per instance in um2."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Leakage per instance in uW."""
        return self.config.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op in ns."""
        return self.config.latency_per_op__ns

    # ----- Value-domain semantics (abstract) -----

    @property
    @abstractmethod
    def x_range(self) -> tuple[int, int]:
        """Inclusive logical integer input range expressible on this tile.

        Single-cycle primitive grid the tile accepts.  Higher-radix
        algorithm-side activations are handled by the macro via
        :class:`~neurox.mapper.Transcoder` and multi-cycle
        accumulation outside the xbar.

        Concrete tiles fix this to their physical primitive grid
        (e.g. ``(0, 1)`` for a binary-input cell).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_count(self) -> int:
        """Number of digits per ``w`` inside this tile.

        Equal to the size of the digit axis in :meth:`fabricate`'s
        input tensor.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_radix(self) -> int:
        """Positional base ``r`` of the in-tile digit combination.

        The tile combines its ``w_digit_count`` digits into one
        logical ``w`` by computing ``sum_k(d_k * r^k)`` inside the
        analog domain.  The derived per-digit weight vector
        ``(1, r, r^2, ..., r^(D-1))`` is materialised internally by
        the concrete xbar's readout where it is needed (e.g. for
        SwitchCap cap-ratio templates); the macro / mapper reads
        this radix directly when building its transcoder.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def w_digit_range(self) -> tuple[int, int]:
        """Inclusive integer range a single digit cell can carry.

        This is the **physical** programmable range of one digit:
        set by the array structure (offset vs differential, ...)
        and the per-cell device-state count, *not* by any signed-
        digit encoding policy.  The macro / mapper's transcoder
        must produce values inside this range; values outside would
        either clip silently (offset tiles) or overflow the per-
        cell state index.

        Example (offset 1T1R array with ``S`` device states and an
        offset shift ``o`` mapping ``digit -> state = digit + o``):
        the digit range is ``(-o, S - 1 - o)`` — typically
        asymmetric and not driven by the encoding's signed-digit
        bound ``(-(r-1), r-1)``.
        """
        raise NotImplementedError

    # ----- Output rescale -----

    @property
    def output_rescale_factor(self) -> float:
        """Active ``(adc_mode, adc_bits) -> rf`` from the externally-built lookup.

        Strict lookup: raises ``KeyError`` whenever the active
        ``(adc_mode, adc_bits)`` row is missing — whether the lookup
        was never populated (no calibration data) or merely lacks the
        requested operating point.  Silently defaulting to ``1.0``
        would corrupt the operator's integer-quantisation pipeline,
        so the absence of a row is treated as a configuration error
        rather than a fallback condition.  Callers that need a valid
        rescale factor must ensure :attr:`XbarConfig.output_rescale_factors`
        carries an entry for the configured runtime operating point.
        """
        key = (self._adc_mode, self._adc_bits)
        return self._rescale_lut[key]

    # ----- Lifecycle -----

    @abstractmethod
    def fabricate(self, w: Tensor) -> None:
        """Program the tile's owned device buffers from an xbar-native digit tensor.

        The tensor is **already** in the xbar's digit grid — the
        macro / mapper layer ran its weight slicing and signed-digit
        transcoding before this call.  The xbar's job is the steps
        that depend on the analog tile: physical-column layout,
        reference-column insertion if any, the device-state offset,
        and forwarding the laid-out tensor to its owned submodules.

        Args:
            w: Integer digit tensor with primitive trailing
                ``[data_num, digit_num, row_num]``, where
                ``digit_num == self.w_digit_count``.  Each entry must
                lie in :attr:`w_digit_range`.  Leading dims are
                broadcast-only — the xbar treats them as opaque
                replica indices.
        """
        raise NotImplementedError

    def _record_xbar_inst_count(self, w: Tensor) -> None:
        """Update the profiled instance count from a fabricated digit tensor.

        ``w`` arrives with primitive trailing
        ``[data_num, digit_num, row_num]``; the number of physical
        xbar replicas is the product of every leading dim before
        those three.  Concrete fabricate implementations should call
        this once per :meth:`fabricate` so the profiler's static
        aggregation reflects the actual replica count.
        """
        leading = w.shape[:-3] if w.ndim >= 3 else ()
        self._record_inst_count(math.prod(leading))

    @abstractmethod
    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Run one analog VMM through the tile.

        Subclasses implement this directly.  Per-cell electrical
        state is read from the xbar's owned device submodules.
        Dynamic energy from every leaf flows through the profiler
        side channel; this method returns numerical output only.

        Concrete implementations may broadcast ``x`` against the
        fabricated per-cell state in whatever order is convenient
        for their analog pipeline.  They are free to insert size-1
        axes internally if their broadcast pattern needs one — the
        caller hands over only the primitive trailing tail
        documented below.

        Args:
            x: Activation tensor with primitive trailing
                ``[row_num]``.  Each entry must lie in
                :attr:`x_range`.  Leading dims are broadcast-only.

        Returns:
            Output tensor with primitive trailing ``[data_num]``.
            Leading shape follows from broadcast between ``x`` and
            the fabricated state.
        """
        raise NotImplementedError
