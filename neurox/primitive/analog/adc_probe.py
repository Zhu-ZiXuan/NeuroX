"""ADC input collection and raw results for independent analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import torch
from torch import Tensor

from neurox.common.base_only_mixin import BaseOnlyMixin
from neurox.common.dataclass_mixin import TensorDataClassMixin, map_single_tensor_fields
from neurox.common.recorder import RecorderBase


class AdcRecord(TensorDataClassMixin, BaseOnlyMixin, ABC, base_only=True):
    # === Public API ===

    def to(self, device: torch.device) -> Self:
        """Return this record with its tensor fields on the requested device.

        Returns:
            `self` when no tensor changes, a record with moved tensors otherwise.
        """
        changed = False

        def move_tensor(tensor: Tensor) -> Tensor:
            nonlocal changed
            moved = tensor.to(device)
            changed = changed or moved is not tensor
            return moved

        moved_record = map_single_tensor_fields(move_tensor, self)
        return moved_record if changed else self

    # === For subclass to implement or override ===

    @abstractmethod
    def input_name(self) -> str:
        """Return the recorded input quantity's unit-bearing name."""
        raise NotImplementedError

    @abstractmethod
    def input_value(self) -> Tensor:
        """Return the scalar decision input at every conversion position."""
        raise NotImplementedError


class IadcRecord(AdcRecord):
    i_in__uA: Tensor
    """Input magnitude current the call was handed."""

    def input_name(self) -> str:
        return "i_in__uA"

    def input_value(self) -> Tensor:
        return self.i_in__uA


class DiffVadcRecord(AdcRecord):
    v_pos__V: Tensor
    """Positive-side input voltage the call was handed."""
    v_neg__V: Tensor
    """Negative-side input voltage the call was handed."""

    def input_name(self) -> str:
        return "v_diff__V"

    def input_value(self) -> Tensor:
        return self.v_pos__V - self.v_neg__V


class AdcProber(RecorderBase[AdcRecord, AdcRecord, list[AdcRecord]]):
    """Keep individual ADC input records in submission order across contexts.

    Records retain their original shapes and types; separate contexts need not
    submit matching counts or layouts.
    """

    @property
    def result(self) -> list[AdcRecord]:
        """Completed records in submission order, shared without copying.

        Consume after collection; further contexts append to the same list.
        """
        return self._history_records

    def submit_current(self, *, i_in__uA: Tensor) -> None:
        """Collect current inputs, preserving their shape and detaching autograd.

        Raises:
            RuntimeError: This instance is not the active ADC prober.
        """
        self._submit_record(IadcRecord(i_in__uA=i_in__uA.detach()))

    def submit_diff_voltage(self, *, v_pos__V: Tensor, v_neg__V: Tensor) -> None:
        """Collect both voltage inputs separately, preserving their shapes.

        Both tensors are detached from autograd. Their difference is computed
        only when a consumer calls the record's `input_value` method.

        Raises:
            RuntimeError: This instance is not the active ADC prober.
        """
        self._submit_record(
            DiffVadcRecord(
                v_pos__V=v_pos__V.detach(),
                v_neg__V=v_neg__V.detach(),
            )
        )

    # === Tools for subclass and internal use ===

    def _merge_records(self, records: Sequence[AdcRecord]) -> Sequence[AdcRecord]:
        return records

    def _sync_history(self, records: Sequence[AdcRecord]) -> Sequence[AdcRecord]:
        if self._sync_device is None:
            return records
        return [record.to(self._sync_device) for record in records]
