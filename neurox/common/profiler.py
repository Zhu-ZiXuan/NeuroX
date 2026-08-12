"""Side-channel dynamic-energy profiler for NeuroX circuit-level simulation."""

import torch
from torch import Tensor

from .recorder import _DEFAULT_DEVICE, RecordBase, RecorderBase


class EnergyRecord(RecordBase):
    """One dynamic-energy record from a physical module's primary execution call.

    Attributes:
        qualified_name: Hierarchical name the emitter carries, as
            :func:`neurox.common.tree.stamp_names` stamped it. A module never
            knows its own name, so a record can only carry one that a walk of
            the assembled model handed out.
        dynamic_energy__fJ: Switching energy attributed to this call. Only the
            caller's leading dims survive, so each element is the energy of one
            unit operation; with no caller leading dims this is a 0-dim scalar.
            Shape: ``[*caller_leading]``.
        channel: Virtual submodule name the emitter passed to
            ``_record_dynamic_energy``; ``None`` for an un-channelled record.
    """

    qualified_name: str
    dynamic_energy__fJ: Tensor
    channel: str | None = None


class Profiler(RecorderBase[EnergyRecord]):
    """Ledger capturing physical modules' dynamic-energy records.

    Every emitter names itself from the stamp its tree gave it, so the model is
    named once after assembly. Read the collected records, or hand them to a
    :class:`neurox.common.reporter.Reporter`, after leaving the context::

        stamp_names(model)
        with Profiler() as profiler:
            model(...)
        print(Reporter(model).render(profiler))

    Energy records follow one repo-wide reduction rule: sum every axis past the
    caller's leading dims, keep the caller's leading dims. Everything after that
    prefix — the emitter's internal work axes (digit, phase, serial round,
    output, instance, …) — is summed, so a record's tensor element is the energy
    of one unit operation rather than a figure already collapsed across the
    batch. A linear unit called with ``[G, T, B, input_num]`` under
    ``leading_rank=3`` therefore yields ``[G, T, B]``. An energy tensor built by
    expanding a constant costs nothing extra: the expanded view holds no storage
    and reducing over its stride-0 axes allocates only the kept prefix.

    ``leading_rank`` is a property of the measurement, not of any module — only
    the measurement site can say how many leading dims its caller owns. It is a
    reporting-resolution knob, not a physical quantity: getting it wrong never
    changes a total (summation is summation, and every scalar view reduces a
    record to a scalar anyway), only how finely the per-unit-operation view
    resolves. There is no runtime check on it.

    Args:
        leading_rank: Number of leading dims the caller owns, ``0`` when the
            measured call has none.
        device: Where a clean exit parks the collected records.
    """

    def __init__(self, *, leading_rank: int = 0, device: torch.device | None = _DEFAULT_DEVICE) -> None:
        if leading_rank < 0:
            raise ValueError(f"leading_rank must be non-negative; got {leading_rank}")
        super().__init__(device=device)
        self._leading_rank = leading_rank

    @property
    def leading_rank(self) -> int:
        """Number of caller leading dims every energy record keeps."""
        return self._leading_rank

    def lay_out(
        self,
        *,
        qualified_name: str,
        dynamic_energy__fJ: Tensor,
        channel: str | None,
    ) -> EnergyRecord:
        """Build one record by folding an energy tensor onto ``[*caller_leading]``.

        The layout rule belongs to the ledger and executes at the emitter, in
        the emitter's own frame: every axis past the caller's leading dims is
        summed, whatever the emitter put there, and the caller's leading dims
        are kept untouched.

        Args:
            qualified_name: The emitter's stamped hierarchical name.
            dynamic_energy__fJ: Per-op dynamic energy as the emitter billed it.
                Shape: ``[*caller_leading, ...]``.
            channel: Virtual submodule name, or ``None``.

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
