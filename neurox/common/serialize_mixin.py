"""Typed mapping, file, and preset serialization for dataclass callers."""

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
    """Load and save dataclass hosts through typed configuration mappings.

    Inherit on a dataclass or a class whose other base applies the dataclass
    transform. Annotated constructor fields define the schema. Loading invokes
    the concrete constructor, including its validation; this mixin does not
    supply a dataclass transform or a separate validation pass.

    Use `from_dict` for an already composed mapping, `from_file` for file
    composition, or `from_preset` for a bundled fragment. Serialization exports
    field values, not module weights or runtime hardware state. Configuration
    classes must be imported before name-based polymorphic dispatch can find
    them.
    """

    # === Public API ===

    @classmethod
    def from_dict(cls, data: Mapping[str, ConfigValue]) -> Self:
        """Construct this dataclass or a selected subclass from a mapping.

        `_neurox_class` may select this type or a known descendant. Nested
        dataclass fields are constructed recursively. Missing required fields
        and unknown keys are rejected. Values must match annotations; integers
        can widen to floats, but booleans do not satisfy integer fields. Enum
        fields use their values, fixed tuples accept matching lists, and paths
        accept strings.

        This method does not expand file-composition directives; use `from_file`
        for mappings containing `_neurox_use` or `_neurox_use_preset`.

        Args:
            data: Composed string-keyed configuration mapping for this
                dataclass.

        Returns:
            A validated instance of this dataclass or the selected concrete
            descendant.

        Raises:
            TypeError: A key, class discriminator, required field, or annotated
                field type is invalid.
            ValueError: Construction fails a value constraint.
        """
        return dataclass_from_dict(cls, data)

    def to_dict(self) -> ConfigDict:
        """Export fields as a fresh serializable configuration mapping.

        Polymorphic dataclasses include their concrete class discriminator.
        Enums and paths use their serializable values, and tuples become lists.
        The output contains configuration values only, not a snapshot of
        tensor-backed module state. Use `to_file` to write the mapping in a
        supported text format.

        Returns:
            A fresh serializable mapping, including discriminators for
            polymorphic dataclasses.
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
        """Compose configuration files and construct a validated dataclass.

        Files are read as TOML or YAML by suffix. Composition directives are
        expanded before section selection in each file. Earlier selected tables
        win conflicts; later tables fill missing keys. A literal dotted key
        takes precedence over traversal through nested tables.

        Args:
            *files: At least one path, resolved from the working directory.
                Relative `_neurox_use` references resolve from their containing
                file.
            section: Table selected from each file, or each file root when
                absent.
            encoding: Text encoding for YAML. TOML always uses UTF-8.
            strict_type: Reject a table/scalar conflict during merge when true.

        Returns:
            A dataclass constructed from the merged selected tables.

        Raises:
            ValueError: No files are supplied, composition or merge is invalid,
                or construction rejects a field value.
            KeyError: A requested section cannot be found.
            TypeError: A section is not a mapping, or construction rejects keys,
                a class discriminator, missing fields, or a field type.
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
        """Construct from a bundled preset path and section.

        The path is relative to installed `neurox/presets`, independent of the
        working directory. A nonempty path and section are required. Absolute
        paths, `./` prefixes, and parent traversal are rejected. Presets may
        compose other presets but cannot load arbitrary external files. Encoding
        and merge options follow `from_file`.

        Args:
            ref: Bundled preset path and section, separated by a colon.
            encoding: Text encoding for YAML fragments; TOML uses UTF-8.
            strict_type: Reject table/scalar conflicts while composing preset
                fragments.

        Returns:
            A dataclass constructed from the selected bundled preset table.
        """
        path, section = parse_preset_ref(ref)
        return cls.from_file(path, section=section, encoding=encoding, strict_type=strict_type)

    def to_file(self, file: Path, *, section: str | None = None, encoding: str = "utf-8") -> None:
        """Write this configuration as TOML or YAML, replacing the target file.

        The suffix selects the format. `section` wraps the exported fields in
        one literal key; dots in that key do not create a nested table path.
        TOML uses UTF-8 and omits mapping entries whose value is `None`; it
        cannot represent `None` inside a list. YAML writes null values and uses
        `encoding`.

        Create the destination directory before calling. This operation exports
        configuration fields; it does not save programmed or fabricated module
        state.

        Args:
            file: Destination path with a supported TOML or YAML suffix.
            section: Optional literal key wrapping the exported mapping.
            encoding: Output encoding for YAML; TOML uses UTF-8.
        """
        data = self.to_dict()
        dict_to_file({section: data} if section is not None else data, file, encoding=encoding)
