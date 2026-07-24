"""The ``SerializeMixin`` object-oriented surface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

from neurox.common.serialize import (
    ConfigDict,
    ConfigValue,
    dataclass_from_dict,
    dataclass_to_dict,
    dict_to_file,
    load_config_dict,
    parse_preset_ref,
)


class SerializeMixin:
    """Add mapping, file, and preset serialization to a dataclass.

    Host requirements:
        - Be a dataclass type.
    """

    @classmethod
    def from_dict(cls, data: Mapping[str, ConfigValue]) -> Self:
        """Build an instance from a plain mapping.

        Args:
            data: Source mapping; every key must match a declared field.

        Returns:
            An instance of ``cls``, or of the concrete subclass ``data`` names.

        Raises:
            TypeError: The mapping cannot be converted to ``cls`` or one of its
                concrete subclasses.
        """
        return dataclass_from_dict(cls, data)

    def to_dict(self) -> ConfigDict:
        """Serialize to a plain nested dict.

        Enums serialize by value, supported containers are converted
        recursively, and polymorphic dataclasses include ``_neurox_class``.

        Returns:
            Nested mapping of serializable values.
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

        Args:
            files: Config file paths (TOML or YAML), ordered by descending
                priority — the first file wins a conflict.
            section: Table to pluck from each file (a dotted name descends
                nested tables); ``None`` takes the file root.
            encoding: YAML text encoding; ignored for TOML.
            strict_type: Reject a dict / non-dict conflict while merging.

        Returns:
            An instance of ``cls``, or of the concrete subclass the files name.

        Raises:
            ValueError: No file was given; at least one is required.
            TypeError: The merged mapping cannot be converted to the target type.
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

        The path is resolved under ``neurox/presets`` and the named section is
        loaded through :meth:`from_file`, preserving its receiver bound.
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
