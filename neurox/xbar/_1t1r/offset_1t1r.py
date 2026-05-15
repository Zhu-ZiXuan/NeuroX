"""Offset-coded 1T1R crossbar tile (analog-tile layout layer).

Architecture (see ``temp/1t1r_xbar.md``)::

    Offset1T1RXbar  ->  Core1T1R  ->  ReadOut [ S/H + WA + MUX + ADC ]

``Offset1T1RXbar`` is the analog-tile layout layer for the offset
weight-encoding scheme:

* signed weight digits live in a single physical sub-array, shifted
  by ``w_state_offset`` so every stored value lands in the RRAM's
  non-negative state-index range;
* each *data* spans ``w_digit_count`` adjacent physical columns
  (data-major, digit-minor);
* one reference column per group of ``ref_group_size`` data is
  programmed to digit ``0`` (state index ``w_state_offset``); the
  analog forward path picks each data's reference and subtracts it
  via the differential ADC.

Layout ownership vs. encoding ownership
---------------------------------------
The xbar receives a **digit tensor** of shape
``[..., data_num, digit_num, row_num]`` whose values are already in
:attr:`w_digit_range`.  Decomposing a high-precision logical weight
into signed digits is the macro / mapper's job (see
:class:`neurox.mapper.xbar.XbarWMapper` and its concrete
:class:`neurox.mapper.xbar.SimpleWMapper`); the xbar owns only the
steps that depend on the analog tile:

1. flatten the ``(data_num, digit_num)`` axes data-major /
   digit-minor onto the physical column axis;
2. insert ref columns through :func:`_insert_ref_cols`;
3. add ``w_state_offset`` so digit ``0`` lands on the RRAM bias
   state (and so every ref cell ends up at the same state);
4. call :meth:`Core1T1R.fabricate` and
   :meth:`ReadOut.fabricate`.

Owned state:

* the **logical-to-physical layout** — ``logic_phys_idx``,
  ``ref_phys_idx``;
* the per-digit analog combination weights ``digit_weights`` buffer
  (LSB-first ``(radix^0, …, radix^(D-1))``) used by the readout's
  data-side SwitchCap to realise the digit weighting passively;
* two owned **submodules** built from factories: ``core`` and
  ``readout``.  The readout bundles ``data_switchcap``,
  ``ref_switchcap``, ``analog_mux`` and ``bl_adc`` into one
  fabricate / readout lifecycle (see ``temp/readout.md``).

:meth:`to_ideal` returns an :class:`IdealXbar` carrying this xbar's
value-domain contract (``x_range``, ``w_digit_count``,
``w_digit_radix``, ``w_digit_range``) and rescale lookup; the ideal
twin discards every encoding detail but accepts the same digit
tensor.

Dynamic metrics
---------------
``vec_mat_mul`` returns only the numerical output tensor.  Dynamic
energy and latency from every leaf (core's DAC / driver / RRAM /
NMOS / OpAmpTIA / wires, readout's data S/H + ref S/H + AnalogMux
+ ADC) flow through the profiler side channel (each leaf is a
:class:`neurox.profiler.ProfiledModule`).
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.analog.readout import ReadOut
from neurox.xbar.base import Xbar, XbarConfig
from neurox.xbar.ideal import IdealXbar

from .core_1t1r import Core1T1R, Core1T1ROutput

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Offset1T1RXbarConfig(XbarConfig):
    """Offset-coded 1T1R xbar configuration.

    Inherits geometry / runtime ADC operating point / rescale lookup
    / PPA from :class:`XbarConfig`.  Only the **offset-coding** knobs
    live here.  Array-physics knobs (wire spacings, WL pulse width)
    live on ``Core1T1RConfig`` because they are independent of the
    weight encoding.

    Attributes:
        w_digit_count: Number of signed digits per data (== number of
            adjacent physical columns each data spans).
        w_digit_radix: Base of the offset-coded per-digit weight.
            :meth:`Offset1T1RXbar.fabricate` materialises the per-digit
            weight vector ``[radix^0, radix^1, ..., radix^(D-1)]``
            (``D == w_digit_count``) and hands it to
            ``ReadOut.fabricate(...)``.  The readout then broadcasts
            it onto the data-side SwitchCap's per-cap ratio tensor —
            the digit weighting is realised passively by the
            bottom-plate-sampled charge-share, not by any algebraic
            ``sum_k(v_k · radix^k)`` outside the leaf circuit modules.
            See :class:`~neurox.analog.readout.OffsetSwitchCapMuxAdcReadOut`.
        w_state_offset: Offset added at fabricate time to convert a
            signed digit value into a non-negative RRAM state index
            (``state = digit + offset``).  The matching ref column,
            holding digit ``0``, ends up at state index ``offset``.
        ref_group_size: Number of *data* (not logic columns) that
            share one ref column.  Must be ``> 0`` and divide
            ``col_num``.
        ref_location: Reference column position *as a data index*
            within each group, 0-indexed from the near-driver side.
            Must satisfy ``0 <= ref_location <= ref_group_size``.
    """

    w_digit_count: int
    w_digit_radix: int
    w_state_offset: int

    ref_group_size: int
    ref_location: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.w_digit_count <= 0:
            raise ValueError(f"require: w_digit_count ({self.w_digit_count}) > 0")
        if self.w_digit_radix <= 1:
            raise ValueError(f"require: w_digit_radix ({self.w_digit_radix}) > 1")
        if self.w_state_offset < 0:
            raise ValueError(f"require: w_state_offset ({self.w_state_offset}) >= 0")
        if self.ref_group_size <= 0:
            raise ValueError(f"require: ref_group_size ({self.ref_group_size}) > 0")
        if self.col_num % self.ref_group_size != 0:
            raise ValueError(f"require: col_num ({self.col_num}) divisible by ref_group_size ({self.ref_group_size})")
        if not (0 <= self.ref_location <= self.ref_group_size):
            raise ValueError(
                f"require: 0 <= ref_location ({self.ref_location}) <= ref_group_size ({self.ref_group_size})"
            )
        if self.adc_bits < 1:
            raise ValueError(f"require: adc_bits ({self.adc_bits}) >= 1")


# ---------------------------------------------------------------------------
# 2. Xbar
# ---------------------------------------------------------------------------


class Offset1T1RXbar(Xbar):
    """Offset-coded 1T1R crossbar tile.

    Owns only the analog-tile layout decisions: digit-major /
    data-minor inlining into physical columns, reference-column
    insertion, the RRAM state-index offset, and the per-digit analog
    combination weight vector consumed by the readout.  Signed-digit
    decomposition of the macro-level weight slice happens upstream
    in :class:`neurox.mapper.xbar.SimpleWMapper` (concrete) or any
    other :class:`neurox.mapper.xbar.XbarWMapper` implementation.

    Owned submodules:
        ``core``: physical core (RRAM + NMOS + OpAmpTIA + wires + DC solver).
        ``readout``: voltage-domain readout chain — data S/H + ref S/H
            + AnalogMux + ADC bundled into one fabricate / readout
            lifecycle.  See ``neurox.analog.readout``.

    Public buffers (built in ``__init__``, addressable by external
    tooling — see ``neurox/tools/xbar_adc_boundaries.py``):
        ``logic_phys_idx``: ``[col_num * w_digit_count]`` long —
            physical-column index of each logic column.
        ``ref_phys_idx``: ``[n_groups]`` long — physical-column index
            of each ref column.
        ``digit_weights``: ``[w_digit_count]`` ``self.dtype`` — the
            offset-code per-digit weight vector
            ``[radix^0, radix^1, ..., radix^(D-1)]``.  Handed to
            :meth:`ReadOut.fabricate` so the data-side SwitchCap's
            per-cap ratios realise the digit weighting passively.

    Args:
        cfg: Offset-coding configuration.
        core_factory: Zero-arg factory returning a fresh
            :class:`Core1T1R`.
        readout_factory: Zero-arg factory returning a fresh
            :class:`ReadOut` (concrete:
            :class:`OffsetSwitchCapMuxAdcReadOut`).
        dtype: Floating-point dtype for the ``digit_weights`` buffer.
    """

    logic_phys_idx: Tensor
    ref_phys_idx: Tensor
    digit_weights: Tensor

    def __init__(
        self,
        cfg: Offset1T1RXbarConfig,
        *,
        name: str = "",
        core_factory: Callable[..., Core1T1R],
        readout_factory: Callable[..., ReadOut],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(cfg, name=name)

        self.cfg = cfg
        self.dtype = dtype

        # Build the two submodules.  Each factory hides its own
        # construction details; the mapping layer treats them as
        # opaque consumers.  Child names cascade from this xbar's
        # hierarchical name so the profiler emits ``<xbar>.core`` /
        # ``<xbar>.readout`` events.
        self.core: Core1T1R = core_factory(name=f"{name}.core" if name else "core")
        self.readout: ReadOut = readout_factory(name=f"{name}.readout" if name else "readout")

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

        Single-digit cell state index is ``state = digit + offset``;
        with ``S`` RRAM device states the cell can carry any state
        in ``[0, S - 1]``, which maps back to a digit range of
        ``(-offset, S - 1 - offset)``.  This is **physical**
        capability — independent of the macro's signed-digit
        encoding policy.  Note that the range is generally
        asymmetric: for the bundled default (``S = 4``,
        ``offset = 1``) it is ``(-1, 2)``, not the symmetric
        ``(-(r-1), r-1)`` a radix-2 signed-digit transcoder would
        naively assume.
        """
        offset = self.cfg.w_state_offset
        states = self.core.w_states
        return (-offset, states - 1 - offset)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def to_ideal(self) -> IdealXbar:
        """Build the noise-free :class:`IdealXbar` twin of this tile.

        The ideal twin inherits this xbar's full value-domain contract
        (``x_range``, ``w_digit_count``, ``w_digit_radix``,
        ``w_digit_range``) and rescale lookup; it carries none of
        the offset encoding.  PPA fields default to zero — the ideal
        tile has no physical cost.  The ideal twin inherits this
        xbar's profiler name so reports keep the same hierarchy.
        """
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

        Steps:

        1. Inline digits into the col axis data-major, digit-minor:
           ``[..., col_num, w_digit_count, row_num]`` ->
           ``[..., col_num * w_digit_count, row_num]``.
        2. Scatter into the physical layout with reference-column
           slots zeroed by :func:`_insert_ref_cols`.
        3. Shift every cell by ``cfg.w_state_offset`` so digit ``0``
           lands on the bias-state RRAM index — the state every ref
           column ends up at.
        4. Hand the physically-shaped tensor to
           :meth:`Core1T1R.fabricate`; the core programs RRAM, samples
           NMOS / OpAmpTIA mismatch, registers every per-cell buffer, and
           binds the DC solver.
        5. Build the grouped readout lattice
           ``(*prefix, group_num, data_num)`` and hand it together
           with :attr:`digit_weights` to :meth:`ReadOut.fabricate`.

        Args:
            w: Xbar-native digit tensor, integer-valued in
                :attr:`w_digit_range`.  Shape:
                ``[..., col_num, w_digit_count, row_num]``.
        """
        # Record the physical xbar instance count for static aggregation.
        # ``prod(w.shape[:-3])`` covers every leading dim (macro batch /
        # M / Tc / Tr / Sa / Sw) per the macro's tile-shape contract.
        self._record_xbar_inst_count(w)
        # Shape: [..., col_num * w_digit_count, row_num] (data-major, digit-minor).
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

    @torch.compile(dynamic=True)
    def vec_mat_mul(self, x: Tensor) -> Tensor:
        """Run one VMM through the core -> readout chain.

        Implements the generic xbar primitive contract — see
        :meth:`Xbar.vec_mat_mul` — so ``x`` need only carry the
        primitive trailing ``[row_num]`` and the output carries the
        primitive trailing ``[data_num]``.  All macro-side leading
        axes (``Bx``, ``M``, ``Tc``, ``Sa``, ``Sw``) ride along as
        opaque broadcast batch dims.

        Implementation detail (1T1R-specific, not part of the generic
        contract): the internal :class:`Core1T1R` broadcasts against
        per-physical-column device buffers, which expect a size-1
        slot at the data position.  This method inserts that slot
        with ``x.unsqueeze(-2)`` before delegating to the core, so
        callers never see the data-broadcast axis.  Dynamic energy
        from every leaf flows through the profiler side channel.

        Steps:

        1. ``core(x.unsqueeze(-2))`` -> per-physical-column OpAmpTIA
           output voltage.
        2. Split ``v_out_phys`` into ``v_data_phys`` and ``v_ref_phys``
           using the layout tables built in ``__init__``; regroup the
           data leg into ``(group_num, data_num, digit_num)``.  Pure
           shape ops.
        3. ``readout(...)`` -> bundled data S/H + ref S/H + AnalogMux
           + differential ADC.
        4. Flatten the code lattice to ``col_num = group_num * data_num``.

        Returns:
            Output digital code with primitive trailing ``[data_num]``.
        """
        # Implementation-specific: re-introduce the size-1 data-
        # broadcast axis the internal Core1T1R expects.
        # Shape: [..., row_num] -> [..., 1, row_num].
        core_out: Core1T1ROutput = self.core(x.unsqueeze(-2))

        v_data_phys = core_out.v_out_phys.index_select(-1, self.logic_phys_idx)
        v_ref_phys = core_out.v_out_phys.index_select(-1, self.ref_phys_idx)
        group_num = self.n_ref_cols
        data_num = self.cfg.ref_group_size
        digit_num = self.cfg.w_digit_count
        v_data_grouped = v_data_phys.unflatten(-1, (group_num, data_num, digit_num))

        readout_out = self.readout.readout(
            v_data_grouped,
            v_ref_phys,
            adc_mode=self._adc_mode,
            adc_bits=self._adc_bits,
        )

        y = readout_out.code.flatten(start_dim=-2)
        return y


# ---------------------------------------------------------------------------
# Reference-column helpers (offset-1T1R-specific layout)
# ---------------------------------------------------------------------------


def _insert_ref_cols(w: Tensor, logic_phys_idx: Tensor, physical_col_num: int) -> Tensor:
    """Scatter logic-column weight digits into the physical-width layout.

    Reference-column slots stay at the zero fill, which combined with
    the caller's downstream ``+ w_state_offset`` shift leaves every
    ref cell programmed to the digit-zero state (RRAM state index
    ``w_state_offset``).  Logic slots receive the caller-supplied
    digit values via ``index_copy_`` along the col axis.

    Args:
        w: Mapped weight digits with the digit axis inlined along the
            col dimension.  Shape:
            ``[..., col_num * w_digit_count, row_num]``.
        logic_phys_idx: Long tensor mapping each logic-column position
            to its physical-column index (built by
            :func:`_build_ref_indices`).
        physical_col_num: Total physical-column count (logic + ref).

    Returns:
        Physical weight digits with ref columns inserted.  Shape:
        ``[..., physical_col_num, row_num]``.
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

    Inverse of :func:`_insert_ref_cols`'s scatter on the trailing
    column axis.  Unit-agnostic — works for any 1D-per-phys-column
    quantity (BL current, OpAmpTIA output voltage, BL clamp voltage, ...).

    Args:
        x: Per-physical-column tensor.  Shape:
            ``[*batch, physical_col_num]``.
        logic_phys_idx: Long tensor mapping each logic-column position
            to its physical index.  Shape: ``[col_num * w_digit_count]``.
        ref_phys_idx: Long tensor mapping each ref-group index to its
            ref column's physical index.  Shape: ``[n_groups]``.

    Returns:
        Tuple ``(x_logic, x_ref)``.  Shapes:
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
    ``group_size_logic`` logic columns plus one ref inserted at
    ``location_logic`` (0-indexed).  The total logic-column count is
    ``n_groups * group_size_logic``; the total physical-column count
    is ``n_groups * (group_size_logic + 1)``.

    For the offset 1T1R xbar the *logical* group size is
    ``ref_group_size * w_digit_count`` and the *logical* ref location
    is ``ref_location * w_digit_count`` — i.e. the data-level
    grouping translated into the digit-inlined column layout.

    Args:
        n_groups: Number of ref groups (== ``col_num // ref_group_size``).
        group_size_logic: Logic columns per ref group
            (== ``ref_group_size * w_digit_count``).
        location_logic: Ref-column position within each group, in
            ``[0, group_size_logic]`` (== ``ref_location * w_digit_count``).

    Returns:
        Tuple of Long tensors ``(logic_phys_idx, ref_phys_idx)`` with
        lengths ``n_groups * group_size_logic`` and ``n_groups``
        respectively.
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
