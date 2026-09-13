"""The `SerializeMixin` object-oriented surface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Self

from .serialize import (
    ConfigDict,
    ConfigValue,
    dataclass_from_dict,
    dataclass_to_dict,
    dict_to_file,
    load_config_dict,
    parse_preset_ref,
)


class SerializeMixin:
    """Provide mapping, file, and preset serialization for dataclass hosts.

    The receiver's annotated fields define the serialization schema. Construction
    uses the host's constructor, including any validation it performs.
    """

    # === Public API ===

    @classmethod
    def from_dict(cls, data: Mapping[str, ConfigValue]) -> Self:
        """Build the receiver or its named concrete subclass from a mapping."""
        return dataclass_from_dict(cls, data)

    def to_dict(self) -> ConfigDict:
        """Convert this dataclass tree into serializable configuration values."""
        return dataclass_to_dict(self)

    @classmethod
    def from_file(
        cls,
        *files: Path,
        section: str | None = None,
        encoding: str = "utf-8",
        strict_type: bool = True,
    ) -> Self:
        """Build the receiver from config files in descending precedence order."""
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
        """Build the receiver from a bundled `family/file:section` preset."""
        path, section = parse_preset_ref(ref)
        return cls.from_file(path, section=section, encoding=encoding, strict_type=strict_type)

    def to_file(self, file: Path, *, section: str | None = None, encoding: str = "utf-8") -> None:
        """Write this dataclass tree to a configuration file selected by suffix."""
        data = self.to_dict()
        dict_to_file({section: data} if section is not None else data, file, encoding=encoding)
