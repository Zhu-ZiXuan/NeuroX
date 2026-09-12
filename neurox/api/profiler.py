"""Side-channel dynamic-energy profiler for NeuroX circuit-level simulation."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.recorder import RecordBase, RecorderBase


class EnergyRecord(RecordBase):
    qualified_name: str
    """Hierarchical name the emitter was stamped with."""
    dynamic_energy__fJ: Tensor
    """Switching energy attributed to this call, one element per unit operation;
    a 0-dim scalar when the caller owns no leading dims.
    Shape: `[*caller_leading]`."""
    channel: str | None
    """Virtual submodule the energy is billed under; `None` for a plain record."""


_Record = EnergyRecord


class Profiler(RecorderBase[_Record]):
    """Ledger capturing physical modules' dynamic-energy records.

    Every emitter names itself from the stamp its tree gave it, so the model is
    named once after assembly. Read the collected records, or hand them to a
    `neurox.Reporter`, after leaving the context:

        stamp_names(model)
        with Profiler() as profiler:
            model(...)
        print(Reporter(model).render(profiler))

    Args:
        leading_rank: Number of leading dims the caller owns, `0` when the
            measured call has none. It belongs to the measurement rather than to
            any module, and sets how finely the per-unit-operation view
            resolves, never a total.
        sync_device: Device a clean exit parks the collected records on; `None`
            leaves each record where it was recorded.
    """

    def __init__(self, *, leading_rank: int = 0, sync_device: torch.device | None = None) -> None:
        if leading_rank < 0:
            raise ValueError(f"leading_rank must be non-negative; got {leading_rank}")
        super().__init__(sync_device=sync_device)
        self._leading_rank = leading_rank

    @property
    def leading_rank(self) -> int:
        return self._leading_rank

    def lay_out(
        self,
        *,
        qualified_name: str,
        dynamic_energy__fJ: Tensor,
        channel: str | None,
    ) -> _Record:
        """Build one record by folding an energy tensor onto `[*caller_leading]`.

        The ledger owns the layout rule and runs it in the emitter's frame:
        every axis past the caller's leading dims is summed, whatever the
        emitter put there — digit, phase, serial round, output, instance — and
        the leading dims are kept untouched. A record's tensor element is
        therefore one unit operation's energy rather than a figure already
        collapsed across the caller-owned leading axes. An input with shape
        `[*caller_leading, ...]` yields `[*caller_leading]`.

        Args:
            qualified_name: The emitter's stamped hierarchical name.
            dynamic_energy__fJ: Dynamic energy as the emitter billed it.
                Shape: `[*caller_leading, ...]`.
            channel: Virtual submodule to bill under, or `None`.
        """
        rank = self._leading_rank
        energy = dynamic_energy__fJ.detach()
        ndim = energy.ndim
        reduced = tuple(range(rank, ndim))
        # Shape: [*caller_leading, ...] -> [*caller_leading]
        if len(reduced) == ndim:
            energy = energy.sum()
        elif reduced:
            energy = energy.sum(dim=reduced)
        return _Record(
            qualified_name=qualified_name,
            dynamic_energy__fJ=energy,
            channel=channel,
        )
