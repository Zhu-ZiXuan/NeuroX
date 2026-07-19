"""The ``SerializeMixin`` OO surface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from neurox.common.serialize import (
    dataclass_from_dict,
    dataclass_to_dict,
    dict_to_file,
    load_config_dict,
    parse_preset_ref,
)


class SerializeMixin:
    """Grant a frozen dataclass the {dict, file} x {read, write} serialization surface.

    ``from_dict`` / ``to_dict`` are structural and format-agnostic. ``from_file`` /
    ``to_file`` add persistence: ``from_file`` layers the given files, resolves the
    ``_neurox_use`` / ``_neurox_use_preset`` directives and the ``_neurox_class``
    discriminator, and returns the (possibly concrete-subclass) instance; ``to_file``
    writes the instance back. ``from_preset`` is a convenience wrapper that loads a
    bundled preset by its ``"family/file:section"`` reference. ``section`` selects one
    top-level table of a multi-config file and is a per-call file-layout argument,
    never a property of the type. The mixin owns no field, runs no validation, and
    holds no state.

    Host requirements:
        - Be a frozen dataclass; every entry point is reflection over the
          declared fields.
    """

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an instance from a plain mapping.

        Nested dataclass and ``Enum`` fields resolve recursively. The receiver
        bounds the result: a ``_neurox_class`` discriminator may name ``cls`` or
        a concrete subclass of it, never a type outside that subtree.

        Args:
            data: Source mapping; every key must match a declared field.

        Returns:
            An instance of ``cls``, or of the concrete subclass ``data`` names.

        Raises:
            TypeError: ``data`` carries a key matching no field of the resolved
                class, names a ``_neurox_class`` outside ``cls``'s subtree, or
                resolves to an abstract base (declared ``ABC`` signal) rather
                than a concrete class.
        """
        return dataclass_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain nested dict.

        ``Enum`` values are written as their ``.value``. A host that belongs to a
        polymorphic family carries a ``_neurox_class`` tag so the round trip
        re-selects the same leaf; a standalone host omits it.

        Returns:
            A nested dict of primitives, ready for :meth:`to_file`.
        """
        return dataclass_to_dict(self)

    @classmethod
    def from_file(
        cls,
        *files: Path,
        section: str | None = None,
        encoding: str = "utf-8",
        strict_type: bool = True,
    ) -> Self:
        """Build an instance from one or more config files.

        Each file is parsed and its ``_neurox_use`` / ``_neurox_use_preset``
        directives are resolved against that file's own directory; the per-file
        results are merged in descending priority, then coerced through
        :meth:`from_dict`, whose receiver bound and subtype validation apply.

        Args:
            files: Config file paths (TOML or YAML), ordered by descending
                priority — the first file wins a conflict.
            section: Top-level table to pluck from each file; ``None`` takes the
                file root.
            encoding: YAML text encoding; ignored for TOML.
            strict_type: Reject a dict / non-dict conflict while merging.

        Returns:
            An instance of ``cls``, or of the concrete subclass the files name.

        Raises:
            ValueError: No file was given; at least one is required.
            TypeError: The merged mapping violates :meth:`from_dict`'s contract.
        """
        return dataclass_from_dict(
            cls,
            load_config_dict(*files, section=section, encoding=encoding, strict_type=strict_type),
        )

    @classmethod
    def from_preset(
        cls,
        ref: str,
        *,
        encoding: str = "utf-8",
        strict_type: bool = True,
    ) -> Self:
        """Build from a bundled preset referenced as ``"family/file:section"``.

        The path is resolved under ``neurox/presets/`` and the named section is
        plucked, then loaded through :meth:`from_file`, inheriting its
        receiver-bounded contract and subtype validation. ``ref`` uses the same
        grammar as the ``_neurox_use_preset`` directive, which reaches the
        identical preset from inside a config file.
        """
        path, section = parse_preset_ref(ref)
        return cls.from_file(path, section=section, encoding=encoding, strict_type=strict_type)

    def to_file(self, file: Path, *, section: str | None = None, encoding: str = "utf-8") -> None:
        """Write this instance to a TOML or YAML file, dispatched by suffix.

        Args:
            file: Destination path; its suffix selects the format.
            section: Top-level table to nest the data under; ``None`` writes it
                at the file root.
            encoding: YAML text encoding; ignored for TOML.

        Raises:
            ValueError: ``file`` carries an unsupported suffix.
        """
        data = self.to_dict()
        dict_to_file({section: data} if section is not None else data, file, encoding=encoding)
