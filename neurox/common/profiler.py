"""Side-channel dynamic-energy profiler for NeuroX circuit-level simulation."""

import torch
from torch import Tensor

from .recorder import RecordBase, RecorderBase


class EnergyRecord(RecordBase):
    """One dynamic-energy record from a physical module's primary execution call."""

    qualified_name: str
    """Hierarchical name the emitter was stamped with."""
    dynamic_energy__fJ: Tensor
    """Switching energy attributed to this call, one element per unit operation;
    a 0-dim scalar when the caller owns no leading dims.
    Shape: `[*caller_leading]`."""
    channel: str | None = None
    """Virtual submodule the energy is billed under; `None` for a plain record."""


class Profiler(RecorderBase[EnergyRecord]):
    """Ledger capturing physical modules' dynamic-energy records.

    Every emitter names itself from the stamp its tree gave it, so the model is
    named once after assembly. Read the collected records, or hand them to a
    `neurox.common.reporter.Reporter`, after leaving the context:

        stamp_names(model)
        with Profiler() as profiler:
            model(...)
        print(Reporter(model).render(profiler))

    Args:
        leading_rank: Number of leading dims the caller owns, `0` when the
            measured call has none. It is a property of the measurement rather
            than of any module — only the measurement site knows how many
            leading dims its caller owns — and a reporting-resolution knob
            rather than a physical quantity: it sets how finely the
            per-unit-operation view resolves, never a total.
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
    ) -> EnergyRecord:
        """Build one record by folding an energy tensor onto `[*caller_leading]`.

        The layout rule belongs to the ledger and executes at the emitter, in
        the emitter's own frame: every axis past the caller's leading dims is
        summed, whatever the emitter put there — digit, phase, serial round,
        output, instance — and the caller's leading dims are kept untouched.
        A record's tensor element is therefore the energy of one unit operation
        rather than a figure already collapsed across the batch: a linear unit
        called with `[G, T, B, input_num]` under `leading_rank=3` yields
        `[G, T, B]`.

        Args:
            qualified_name: The emitter's stamped hierarchical name.
            dynamic_energy__fJ: Dynamic energy as the emitter billed it.
                Shape: `[*caller_leading, ...]`.
            channel: Virtual submodule to bill under, or `None`.

        Returns:
            The finished record, its energy folded onto the caller's block.
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
        return EnergyRecord(
            qualified_name=qualified_name,
            dynamic_energy__fJ=energy,
            channel=channel,
        )
