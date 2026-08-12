"""Aggregation, naming, and canonical presentation of one model's metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from .profile_mixin import ProfileMixin
from .profiler import EnergyRecord, Profiler


@dataclass(frozen=True)
class StaticEntry:
    """One static row: a profile target's fabrication-time metrics.

    Attributes:
        qualified_name: Hierarchical name, as the bound model names the module.
        area__um2: Module-local total area = ``area_per_inst * inst_count``.
        leakage__uW: Module-local leakage power total.
    """

    qualified_name: str
    area__um2: float
    leakage__uW: float


@dataclass(frozen=True)
class DynamicEntry:
    """One dynamic row: the energy every record sharing a path adds up to.

    Attributes:
        path: Hierarchical name segments, a virtual channel included as its own
            trailing segment. The bound model itself is named by the empty
            string, so it has no segments of its own and its virtual children
            read with a leading empty one.
        dynamic_energy__fJ: Energy summed over the merged records and over every
            unit operation each of them holds.
    """

    path: tuple[str, ...]
    dynamic_energy__fJ: float

    @property
    def qualified_name(self) -> str:
        """The path as one dotted string — the canonical row key."""
        return ".".join(self.path)


@dataclass(frozen=True)
class StaticMetrics:
    """Static hardware totals over one model.

    Attributes:
        area__um2: Area summed across every profile target.
        leakage__uW: Leakage power summed across every profile target.
    """

    area__um2: float
    leakage__uW: float


class Reporter:
    """Turn one model's modules and one profiler's records into report rows.

    A record carries the name its emitter was stamped with, so the reporter
    binds one model at construction and walks it once: to collect the static
    rows, and to check that every profile-capable module it holds is stamped
    with the name this very walk gives it. The walk keeps every duplicate path,
    exactly as the stamping walk does, so an instance bound at a second location
    is met under a name its stamp cannot match. Building the reporter before the
    measurement is therefore the canonical order — the walk is where a missing
    or stale stamp is caught, well before any record is read. Static rows follow
    from the walk alone; the dynamic views take the profiler holding the records
    as an argument, so one reporter serves any number of measurements of the
    same model.

    Args:
        model: The tree every reported name is resolved against.

    Raises:
        ValueError: A profile-capable module of ``model`` carries no name stamp,
            carries one from another tree, or is bound at a second location.
    """

    def __init__(self, model: nn.Module) -> None:
        module_names: set[str] = set()
        static_entries: list[StaticEntry] = []
        for name, module in model.named_modules(remove_duplicate=False):
            module_names.add(name)
            if not isinstance(module, ProfileMixin):
                continue
            try:
                stamped = module.qualified_name
            except RuntimeError:
                raise ValueError(
                    f"module {name!r} carries no name stamp; "
                    "call stamp_names(model) once the model is assembled, before reporting it"
                ) from None
            if stamped != name:
                raise ValueError(
                    f"module {name!r} is stamped {stamped!r}; its stamp is stale or belongs to another tree, "
                    "or one instance is bound at both locations, so re-run stamp_names(model) on the model "
                    "being reported and bind one instance per location"
                )
            if module.is_profile_target:
                static_entries.append(
                    StaticEntry(
                        qualified_name=name,
                        area__um2=module.area__um2,
                        leakage__uW=module.leakage__uW,
                    )
                )
        self._module_names = frozenset(module_names)
        self._static_entries = tuple(static_entries)
        self._static = StaticMetrics(
            area__um2=sum((entry.area__um2 for entry in static_entries), 0.0),
            leakage__uW=sum((entry.leakage__uW for entry in static_entries), 0.0),
        )

    @property
    def static_entries(self) -> tuple[StaticEntry, ...]:
        """Per-module static rows, in the bound model's traversal order."""
        return self._static_entries

    @property
    def static(self) -> StaticMetrics:
        """Static totals over every profile target the bound model holds."""
        return self._static

    def dynamic_entries(self, profiler: Profiler) -> tuple[DynamicEntry, ...]:
        """Merge the profiler's records into one row per path.

        Args:
            profiler: The ledger holding the records to report.

        Returns:
            One row per distinct path, ordered by descending energy.

        Raises:
            ValueError: A record cannot be named against the bound model; see
                :meth:`by_name`.
        """
        records = profiler.records
        energies = self._scalars__fJ(records)
        merged: dict[tuple[str, ...], float] = {}
        for record, energy in zip(records, energies, strict=True):
            path = self._path_of(record)
            merged[path] = merged.get(path, 0.0) + energy
        entries = [DynamicEntry(path=path, dynamic_energy__fJ=energy) for path, energy in merged.items()]
        entries.sort(key=lambda entry: entry.dynamic_energy__fJ, reverse=True)
        return tuple(entries)

    def by_name(self, profiler: Profiler) -> dict[str, float]:
        """Group the profiler's dynamic energy by qualified row name [fJ].

        A channelled record groups under ``"<module dotted name>.<channel>"``
        instead of the bare module name, so a composite's distinct billed
        branches appear as separate rows.

        Args:
            profiler: The ledger holding the records to report.

        Returns:
            Row name to its energy total, in first-emission order.

        Raises:
            ValueError: A record names a module the bound model does not hold, a
                channel carries a dot, or a channel's virtual name collides with
                a real module of the bound model.
        """
        records = profiler.records
        by_name: dict[str, float] = {}
        for record, energy in zip(records, self._scalars__fJ(records), strict=True):
            name = ".".join(self._path_of(record))
            by_name[name] = by_name.get(name, 0.0) + energy
        return by_name

    def by_group(self, profiler: Profiler, groups: Mapping[str, str]) -> dict[str, float]:
        """Group the profiler's dynamic energy by a caller-supplied label [fJ].

        The grouping is stated over row names — the very vocabulary
        :meth:`by_name` returns, virtual channel rows included — so a caller
        folds any set of rows into one figure without the reporter guessing
        what belongs together. Every measured row must be mapped: a row the
        grouping does not cover is an omission the reporter refuses to hide.
        A mapped row that no record used contributes nothing and is not
        reported, so one grouping policy may cover more rows than a given
        measurement exercises.

        Args:
            profiler: The ledger holding the records to report.
            groups: Qualified row name to the label it contributes to.

        Returns:
            Label to its energy total, in first-contribution order.

        Raises:
            ValueError: A measured row is absent from ``groups``, or a record
                cannot be named against the bound model; see :meth:`by_name`.
        """
        records = profiler.records
        by_group: dict[str, float] = {}
        for record, energy in zip(records, self._scalars__fJ(records), strict=True):
            name = ".".join(self._path_of(record))
            if name not in groups:
                raise ValueError(
                    f"row {name!r} is not covered by the grouping; "
                    "map every row the measurement produces, listing them with by_name()"
                )
            label = groups[name]
            by_group[label] = by_group.get(label, 0.0) + energy
        return by_group

    def total_dynamic_energy__fJ(self, profiler: Profiler) -> float:
        """Sum the profiler's dynamic energy over every record and unit operation.

        Args:
            profiler: The ledger holding the records to report.

        Returns:
            The measurement's whole dynamic energy [fJ].
        """
        return sum(self._scalars__fJ(profiler.records), 0.0)

    def render(self, profiler: Profiler | None = None) -> str:
        """Render the canonical inspection dump.

        Units are fixed — fJ, um2, uW — and never auto-scaled, so two dumps are
        comparable line by line. With no profiler the dump is the static table
        alone.

        Args:
            profiler: The ledger holding the records to report, or ``None`` for
                a static-only dump.

        Returns:
            Aligned plain-text tables, one blank line between them.
        """
        sections: list[str] = []
        if profiler is not None:
            entries = self.dynamic_entries(profiler)
            rows = [[entry.qualified_name, f"{entry.dynamic_energy__fJ:.4f}"] for entry in entries]
            total__fJ = sum((entry.dynamic_energy__fJ for entry in entries), 0.0)
            rows.append(["total", f"{total__fJ:.4f}"])
            table = _aligned_table(["name", "energy [fJ]"], rows, text_columns=1)
            sections.append("\n".join(["dynamic", *table]))
        rows = [
            [entry.qualified_name, f"{entry.area__um2:.4f}", f"{entry.leakage__uW:.4f}"]
            for entry in self._static_entries
        ]
        rows.append(["total", f"{self._static.area__um2:.4f}", f"{self._static.leakage__uW:.4f}"])
        table = _aligned_table(["name", "area [um2]", "leakage [uW]"], rows, text_columns=1)
        sections.append("\n".join(["static", *table]))
        return "\n\n".join(sections)

    @staticmethod
    def _scalars__fJ(records: Sequence[EnergyRecord]) -> list[float]:
        """Total each record's energy tensor to a host float in one device sync.

        A record keeps its per-unit-operation layout on the device it was parked
        on; every scalar view reduces through here, so one aggregation call
        costs one host transfer however many records it covers.

        Args:
            records: Records to reduce.

        Returns:
            One total per record, positionally aligned with ``records``.
        """
        if not records:
            return []
        return torch.stack([record.dynamic_energy__fJ.sum() for record in records]).cpu().tolist()

    def _path_of(self, record: EnergyRecord) -> tuple[str, ...]:
        """Resolve one record's row path against the bound model.

        A channel is a virtual submodule: a branch its parent bills without a
        module instance of its own. Its row name is the emitter's name with the
        channel appended — ``f"{module_name}.{channel}"`` — and it must be able
        to be one segment: no dot inside it, no name a real child of the bound
        model already holds. The bound model's own name is empty, so its virtual
        children carry a leading empty segment (``".cablc"``), which is what
        marks a branch the model bills itself apart from a top-level child of
        the same name.

        The record's own name is checked against the walk as a backstop: the
        stamp gate at construction covers every module the model holds now, and
        this catches a name no longer among them — a record from a differently
        shaped tree, or from a module the model has been rewired to drop.

        Args:
            record: The record to name.

        Returns:
            The row's name segments; empty for the bound model's own
            un-channelled records.

        Raises:
            ValueError: The record names a module outside the bound model, the
                channel carries a dot, or the virtual name collides with a real
                module.
        """
        name = record.qualified_name
        if name not in self._module_names:
            raise ValueError(
                f"a record is named {name!r}, which the reported model does not hold; "
                "report against a model that holds it, and select subtrees by filtering records"
            )
        channel = record.channel
        if channel is None:
            return tuple(name.split(".")) if name else ()
        if "." in channel:
            raise ValueError(f"channel {channel!r} on module {name!r} names one virtual submodule, not a path")
        virtual = f"{name}.{channel}"
        if virtual in self._module_names:
            raise ValueError(f"channel {channel!r} on module {name!r} collides with the real module {virtual!r}")
        return tuple(virtual.split("."))


def _aligned_table(header: Sequence[str], rows: Sequence[Sequence[str]], *, text_columns: int) -> list[str]:
    """Lay one table out as column-aligned lines, its header line first.

    Args:
        header: Column titles.
        rows: Pre-formatted cells, one sequence per row, each as long as
            ``header``.
        text_columns: Number of leading columns to left-align; the rest are
            right-aligned as numbers.

    Returns:
        The header line followed by one line per row.
    """
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    lines: list[str] = []
    for row in (header, *rows):
        cells = [
            cell.ljust(width) if index < text_columns else cell.rjust(width)
            for index, (cell, width) in enumerate(zip(row, widths, strict=True))
        ]
        lines.append("  ".join(cells).rstrip())
    return lines
