"""Side-channel tensor prober for calibration and diagnostics.

See also:
    docs/internals/common/prober.md
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import ClassVar, Self

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import ProbeMixin

# One probe record: the emitting module plus the named tensors it submitted.
ProbeRecord = tuple[ProbeMixin, dict[str, Tensor]]


class Prober:
    """Context manager that captures per-call tensor records on named channels.

    Canonical usage — record inside the ``with`` block, read the channel
    accessors after it::

        with Prober(channels=frozenset({"some.channel"})) as prober:
            model(...)
        for module, tensors in prober.records("some.channel"):
            ...

    Probers stack: entering pushes onto a class-level active stack, exiting
    pops. An emission reaches every stacked prober, so an outer
    session-scoped prober keeps recording while an inner one captures a
    narrow window. Each prober applies its own channel allowlist;
    ``channels=None`` accepts every channel.

    A record carries its emitting module, not a name: a module never knows
    its own name, so only a walk from a root can name it —
    :meth:`resolve_names` builds that map read-only. Tensors are stored
    detached, on their recording device; no sync or copy happens at
    submission time.

    Unlike the profiler's energy path, records are not reduced: a probe
    stores full tensors for downstream fitting / diagnostics, so a probing
    run holds every submitted tensor alive until the prober is dropped.
    """

    # Class-level LIFO stack of active probers; emissions reach every member.
    _active_stack: ClassVar[list[Prober]] = []

    def __init__(self, *, channels: frozenset[str] | None = None) -> None:
        self._channels = channels
        self._records: dict[str, list[ProbeRecord]] = {}

    def __enter__(self) -> Self:
        Prober._active_stack.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        popped = Prober._active_stack.pop()
        if popped is not self:
            raise RuntimeError("Prober exited out of LIFO order; the active stack is corrupted")

    @property
    def channels(self) -> frozenset[str] | None:
        """Channel allowlist; ``None`` accepts every channel."""
        return self._channels

    # ----------------------------------------------------------------
    # Side-channel entry point (called by ProbeMixin._probe_record)
    # ----------------------------------------------------------------

    def submit(self, channel: str, module: ProbeMixin, tensors: Mapping[str, Tensor]) -> None:
        """Store one detached record on ``channel`` if the allowlist admits it.

        Args:
            channel: Probe channel name the emitter submitted on.
            module: The emitting module; stored by identity, never named.
            tensors: Named per-call tensors; stored detached, in
                submission order.
        """
        if self._channels is not None and channel not in self._channels:
            return
        self._records.setdefault(channel, []).append((module, {k: v.detach() for k, v in tensors.items()}))

    # --------------------------- Accessors ---------------------------

    def records(self, channel: str) -> list[ProbeRecord]:
        """Submission-ordered ``(module, tensors)`` records on ``channel``."""
        return list(self._records.get(channel, []))

    def stacked(self, channel: str, key: str) -> Tensor:
        """Stack one named tensor across every record on ``channel``.

        Args:
            channel: Probe channel to read.
            key: Tensor name within each record; every record must carry
                it at a common shape.

        Returns:
            Tensor with a new leading record axis, in submission order.
        """
        recs = self._records.get(channel, [])
        if not recs:
            raise ValueError(f"no records on channel {channel!r}")
        return torch.stack([tensors[key] for _, tensors in recs])

    def paired(self, ch_a: str, ch_b: str) -> list[tuple[ProbeRecord, ProbeRecord]]:
        """Order-aligned record pairs from two channels.

        Alignment is positional: record ``i`` of ``ch_a`` pairs with
        record ``i`` of ``ch_b``, so the two channels must have been fed
        by the same call sequence.

        Raises:
            ValueError: If the two channels hold different record counts.
        """
        recs_a = self._records.get(ch_a, [])
        recs_b = self._records.get(ch_b, [])
        if len(recs_a) != len(recs_b):
            raise ValueError(f"paired({ch_a!r}, {ch_b!r}): record counts differ ({len(recs_a)} vs {len(recs_b)})")
        return list(zip(recs_a, recs_b, strict=True))

    @staticmethod
    def resolve_names(root: nn.Module) -> dict[int, str]:
        """Read-only ``id(module) -> qualified name`` map from ``root``.

        The map keys on ``id`` so a stored emitter is named without
        assuming it is hashable-by-identity elsewhere; the root itself is
        named ``""``, as its own traversal names it.
        """
        return {id(module): name for name, module in root.named_modules()}


class AdcProber(Prober):
    """Prober pre-scoped to the ADC calibration channels.

    ``adc.convert`` carries every physical ADC conversion (inputs, code,
    operating-point fields as the emitting family defines); ``adc.ideal_vmm``
    carries the ideal tile's codes. Pairing the two streams from a
    physical and an ideal run of the same stimulus is the calibration view a
    rescale fit consumes.
    """

    ADC_CONVERT: ClassVar[str] = "adc.convert"
    ADC_IDEAL_VMM: ClassVar[str] = "adc.ideal_vmm"

    def __init__(self, *, channels: frozenset[str] | None = None) -> None:
        if channels is None:
            channels = frozenset({AdcProber.ADC_CONVERT, AdcProber.ADC_IDEAL_VMM})
        super().__init__(channels=channels)

    # ---------------------- Calibration views -----------------------

    def convert_records(self) -> list[ProbeRecord]:
        """Submission-ordered records on :data:`ADC_CONVERT`."""
        return self.records(AdcProber.ADC_CONVERT)

    def ideal_vmm_records(self) -> list[ProbeRecord]:
        """Submission-ordered records on :data:`ADC_IDEAL_VMM`."""
        return self.records(AdcProber.ADC_IDEAL_VMM)

    def paired_conversions(self) -> list[tuple[ProbeRecord, ProbeRecord]]:
        """Order-aligned ``(convert, ideal_vmm)`` record pairs."""
        return self.paired(AdcProber.ADC_CONVERT, AdcProber.ADC_IDEAL_VMM)
