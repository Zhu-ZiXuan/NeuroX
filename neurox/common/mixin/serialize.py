"""The ``SerializeMixin`` OO surface.

See also:
    docs/internals/common/mixin/serialize.md
"""

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
    """

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return dataclass_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_to_dict(self)

    @classmethod
    def from_file(
        cls,
        *files: Path,
        section: str | None = None,
        encoding: str = "utf-8",
        strict_type: bool = True,
    ) -> Self:
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
        receiver-bounded contract and subtype validation.
        """
        path, section = parse_preset_ref(ref)
        return cls.from_file(path, section=section, encoding=encoding, strict_type=strict_type)

    def to_file(self, file: Path, *, section: str | None = None, encoding: str = "utf-8") -> None:
        data = self.to_dict()
        dict_to_file({section: data} if section is not None else data, file, encoding=encoding)
