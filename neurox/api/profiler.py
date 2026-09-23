"""Collect and expose hardware snapshots and execution observations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from typing import Literal, cast
from weakref import WeakValueDictionary

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.dataclass_mixin import TensorDataClassMixin
from neurox.common.module import ProfileModule, neurox_profile_modules
from neurox.common.recorder import RecorderBase

from .function import stamp_names

_source_ids = count()
_sources: WeakValueDictionary[int, ProfileModule] = WeakValueDictionary()


@dataclass(eq=False, kw_only=True)
class ProfileItem:
    area__um2: float | None
    """Local area of all instances, excluding children; `None` when unavailable."""
    leakage__uW: float | None
    """Local leakage of all instances, excluding children; `None` when unavailable."""
    dynamic_energy__fJ: Tensor | None
    """Energy concatenated in collection order along the configured axis;
    all other sample axes are preserved. `None` when no energy was submitted.
    Shape: `[*measurement]`."""
    working_duration__ns: Tensor | None
    """This name's concatenated basic-operation durations over sample positions;
    `None` when no timing was submitted. Ancestor timing is not inherited here.
    Shape: `[*measurement]`."""


@dataclass(frozen=True, kw_only=True)
class _StaticItem:
    area__um2: float
    """Local area of all instances, excluding children."""
    leakage__uW: float
    """Local leakage of all instances, excluding children."""


@dataclass(eq=False, kw_only=True)
class _DynamicItem:
    dynamic_energy__fJ: Tensor | None = None
    """One context's combined contributions, preserving the observation layout;
    `None` when this name received no energy submission.
    Shape: `[*measurement]`."""
    working_duration__ns: Tensor | None = None
    """Basic-operation durations over this context's sample positions; `None`
    when no latency was submitted. Child costs are not inferred.
    Shape: `[*measurement]`."""


type _History = dict[str, _DynamicItem]


class _EnergyRecord(TensorDataClassMixin):
    name: str
    channel: str | None
    dynamic_energy__fJ: Tensor


class _LatencyRecord(TensorDataClassMixin):
    name: str
    latency__ns: Tensor


class Profiler(RecorderBase[_EnergyRecord | _LatencyRecord, _History, dict[str, ProfileItem]]):
    """Snapshot hardware costs and collect named execution data per context.

    `collect_static_data` names an assembled model and snapshots local area and
    leakage after physical-state setup. For dynamic-only collection, name the
    model with `stamp_names` instead. Keep names and hardware state fixed across
    the measurement. Each completed context retains only submitted names;
    an empty context is retained too.

    Each energy element represents one basic operation of its owning unit. The
    model owner configures module profile ranks to preserve those positions.
    Emitters reduce internal energy contributions and submit matching operation
    durations. Each unit contributes one completed operation per model-batch
    context, aggregating its phases and numerical chunks before submission;
    distinct operator positions own distinct units.

    Scalar energy and duration submissions represent one observation and gain
    a length-one sample axis before export. Existing observation axes remain
    unchanged. Even a single context therefore exports scalars with shape `[1]`.

    Completed energy and duration tensors always reside on CPU, independently
    of the model's device and PyTorch's default device. Snapshots are exported
    when submitted and ready before context-exit aggregation. CUDA-to-CPU
    exports use a separate stream.

    Args:
        concat_dim: Fixed export concatenation axis for all names and tensor
            fields after scalar normalization. Other axes must match across
            contexts; concatenation does not reduce or insert axes.
        on_repeat: `"sum"` adds same-name energy contributions elementwise,
            requiring equal shapes; `"replace"` retains the last contribution.
            This policy does not combine independent unit calls or sum time.

    Raises:
        ValueError: Context-exit validation finds incompatible summed shapes or
            repeated unit timing submissions.
    """

    def __init__(
        self,
        *,
        concat_dim: int,
        on_repeat: Literal["sum", "replace"] = "sum",
    ) -> None:
        super().__init__()
        self._concat_dim = concat_dim
        self._on_repeat = on_repeat
        self._static_data: dict[str, _StaticItem] = {}

    # === Public API ===

    @staticmethod
    def register_source(source: ProfileModule) -> Tensor:
        """Bind a module to a weakly held, runtime-resolved name source.

        Call outside compilation during construction or after copying. Keep
        the returned scalar int64 identity on CPU as ordinary host metadata;
        it is independent of module device placement. Submission resolves the
        module's current stamped name without specializing on that string.
        """
        identity = next(_source_ids)
        _sources[identity] = source
        return torch.tensor(identity, dtype=torch.int64, device="cpu")

    @property
    def on_repeat(self) -> Literal["sum", "replace"]:
        return self._on_repeat

    @property
    def result(self) -> dict[str, ProfileItem]:
        """Validate completed observations and concatenate their named tensors.

        Each access traverses history and concatenates multiple contexts into
        new tensors; a single context can share its tensors. Static costs are
        included once, and only directly submitted durations are exported.

        Energy and timing submitted under the same name must describe the same
        sample layout in each context. Names are handled independently; absent
        timing remains `None`.

        Raises:
            ValueError: Contexts disagree on names or field presence, or energy
                and timing submitted under the same name have different layouts.
            RuntimeError: Tensor layouts cannot be concatenated.
        """
        history = self._history_records

        # --- 1: initialize results from the static snapshot ---

        data = {
            name: ProfileItem(
                area__um2=static.area__um2,
                leakage__uW=static.leakage__uW,
                dynamic_energy__fJ=None,
                working_duration__ns=None,
            )
            for name, static in self._static_data.items()
        }

        if not history:
            return data

        # --- 2: validate names, field presence, and sample layouts ---

        first_batch = history[0]
        for batch_index, batch in enumerate(history):
            if batch.keys() != first_batch.keys():
                raise ValueError(f"batch {batch_index} has different profile names from batch 0")
            for name, batch_item in batch.items():
                first_batch_item = first_batch[name]
                energy = batch_item.dynamic_energy__fJ
                duration = batch_item.working_duration__ns
                if (energy is None) != (first_batch_item.dynamic_energy__fJ is None):
                    raise ValueError(f"dynamic_energy__fJ for {name!r} mixes None and values at batch {batch_index}")
                if (duration is None) != (first_batch_item.working_duration__ns is None):
                    raise ValueError(f"working_duration__ns for {name!r} mixes None and values at batch {batch_index}")
                if energy is not None and duration is not None and energy.shape != duration.shape:
                    raise ValueError(
                        f"energy and timing for {name!r} have different sample shapes at batch {batch_index}"
                    )

        # --- 3: concatenate and fill the submitted fields ---

        for name in first_batch:
            energies = [energy for batch in history if (energy := batch[name].dynamic_energy__fJ) is not None]
            durations = [duration for batch in history if (duration := batch[name].working_duration__ns) is not None]

            all_energy = None
            if energies:
                all_energy = energies[0] if len(energies) == 1 else torch.cat(energies, dim=self._concat_dim)

            all_duration = None
            if durations:
                all_duration = durations[0] if len(durations) == 1 else torch.cat(durations, dim=self._concat_dim)

            profile_item = data.get(name)
            if profile_item is None:
                profile_item = ProfileItem(
                    area__um2=None,
                    leakage__uW=None,
                    dynamic_energy__fJ=None,
                    working_duration__ns=None,
                )
                data[name] = profile_item
            profile_item.dynamic_energy__fJ = all_energy
            profile_item.working_duration__ns = all_duration
        return data

    def collect_static_data(self, model: nn.Module) -> None:
        """Name the model and snapshot local area and leakage totals.

        Call after physical-state setup and outside collection. Names are
        relative to `model`, including ordinary PyTorch containers. The model
        is not retained. History is unchanged and must correspond to the same
        hardware state and names.

        Raises:
            ValueError: A NeuroX module is bound at multiple model paths.
        """
        stamp_names(model)
        self._static_data = {
            module.qualified_name: _StaticItem(area__um2=module.area__um2, leakage__uW=module.leakage__uW)
            for _, module in neurox_profile_modules(model)
        }

    @RecorderBase.submission
    def submit_dynamic_energy(
        self,
        dynamic_energy__fJ: Tensor,
        *,
        name: str | None = None,
        channel: str | None = None,
        source: Tensor | None = None,
    ) -> None:
        """Buffer energy already reduced to the emitter's observation axes.

        Retain a detached copy without changing dtype, exported to CPU.
        CPU exports from CUDA become readable at context exit. Names passed
        explicitly here specialize compiled callers; module emission resolves
        names at runtime.

        Args:
            dynamic_energy__fJ: Contributions for corresponding basic operations.
                No additional axes are reduced. A scalar gains one sample axis.
                Shape: `[*measurement]`.
            name: Emitter's full stamped module path; empty for the named root.
                Supply exactly one of `name` and `source`.
            channel: Optional billing segment under the emitter; a matching
                real child receives the contribution in its existing row.
            source: CPU identity from `register_source`, resolved at runtime.

        Raises:
            ValueError: The supplied channel is empty or contains a dot, or
                exactly one of `name` and `source` was not supplied.
        """
        if channel is not None and (not channel or "." in channel):
            raise ValueError(f"channel {channel!r} must name one nonempty virtual segment")
        if dynamic_energy__fJ.ndim == 0:
            dynamic_energy__fJ = dynamic_energy__fJ.unsqueeze(0)
        self._submit_record(
            _EnergyRecord(
                name=self._resolve_name(name, source),
                channel=channel,
                dynamic_energy__fJ=self._export_tensor(dynamic_energy__fJ),
            )
        )

    @RecorderBase.submission
    def submit_latency(self, latency__ns: Tensor, *, name: str | None = None, source: Tensor | None = None) -> None:
        """Buffer detached copies of basic-operation durations with their sample layout.

        Each position matches one retained energy position. Expanded views are
        supported; no axes are reduced. Scalars gain one sample axis. Repeated
        submissions from one unit fail at context exit.

        Args:
            latency__ns: Finite, nonnegative floating-point durations, exported
                to CPU and ready for aggregation at context exit.
                Shape: `[*measurement]`.
            name: Emitter's full stamped module path; empty for the named root.
                Supply exactly one of `name` and `source`.
            source: CPU identity from `register_source`, resolved at runtime.
        """
        if latency__ns.ndim == 0:
            latency__ns = latency__ns.unsqueeze(0)
        self._submit_record(
            _LatencyRecord(
                name=self._resolve_name(name, source),
                latency__ns=self._export_tensor(latency__ns),
            )
        )

    # === Tools for subclass and internal use ===

    @staticmethod
    def _resolve_name(name: str | None, source: Tensor | None) -> str:
        if (name is None) == (source is None):
            raise ValueError("supply exactly one of name and source")
        if name is not None:
            return name
        return _sources[int(cast(Tensor, source).item())].qualified_name

    def _merge_records(self, records: Sequence[_EnergyRecord | _LatencyRecord]) -> Sequence[_History]:
        items: _History = {}

        # --- Combine submitted names with channels; merge this context's submissions ---

        for record in records:
            name = record.name
            if isinstance(record, _EnergyRecord) and record.channel is not None:
                channel = record.channel
                name = f"{name}.{channel}" if name else channel
            item = items.get(name)
            if item is None:
                item = _DynamicItem()
                items[name] = item
            if isinstance(record, _LatencyRecord):
                if item.working_duration__ns is not None:
                    raise ValueError(f"unit {name!r} submitted its working duration more than once in one batch")
                item.working_duration__ns = record.latency__ns
            else:
                energy = record.dynamic_energy__fJ
                previous = item.dynamic_energy__fJ
                if self.on_repeat == "sum" and previous is not None:
                    if previous.shape != energy.shape:
                        raise ValueError(f"cannot sum different measurement shapes for {name!r}")
                    energy = previous + energy
                item.dynamic_energy__fJ = energy

        return [items]
