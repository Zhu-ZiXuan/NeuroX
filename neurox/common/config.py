"""Config base classes with TOML/YAML I/O and builder pattern.

``Config`` is the mixin base for all NeuroX runtime configuration
dataclasses.  It provides ``from_file`` / ``to_file`` for transparent
TOML and YAML serialization and a ``from_builder`` bridge to the
``ConfigBuilder`` pattern.  A single file may hold several configs as
named top-level tables; pass ``section=`` to pluck one out.

``ConfigBuilder`` is a generic base for staged config construction with
explicit validation.  Builders are also dataclasses, hold optional
fields, and implement ``validate`` + ``build`` to produce a ``Config``
instance.  They share the same ``from_file`` / ``to_file`` interface so
builder state can be persisted and loaded independently.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Self, TypeVar

from .load_dump import dataclass_from_file, dataclass_to_file

C = TypeVar("C", bound="Config")
B = TypeVar("B", bound="ConfigBuilder[Config]")


class ValidationError(Exception):
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        msg = "Multiple validation errors:\n" + "\n".join(f"  - {type(e).__name__}: {e}" for e in errors)
        super().__init__(msg)


class Config:
    """Base class for NeuroX runtime configuration objects.

    Concrete configs are dataclasses that inherit this class to gain
    TOML/YAML serialization and the ``from_builder`` factory::

        @dataclass
        class MyConfig(Config):
            foo: str
            bar: int
    """

    # --- file io --- #

    @classmethod
    def from_file(
        cls,
        *files: Path,
        section: str | None = None,
        encoding: str | None = "utf-8",
        strict_type: bool = True,
    ) -> Self:
        """Build this config from one or more TOML/YAML files.

        Args:
            files: Config file paths ordered by descending priority.
                When several files are given they are deep-merged so a
                user override can layer on top of a default file.
            section: Optional top-level table name to pluck out of each
                file.  Use this when several configs share one file.
            encoding: Text encoding for YAML (ignored for TOML).
            strict_type: Reject dict/non-dict conflicts during merge.
        """
        return dataclass_from_file(
            cls,
            *files,
            section=section,
            encoding=encoding,
            strict_type=strict_type,
        )

    def to_file(self, file: Path, *, encoding: str | None = "utf-8") -> None:
        """Dump this config to a TOML or YAML file."""
        dataclass_to_file(self, file, encoding=encoding)

    # --- builder bridge --- #

    @classmethod
    def from_builder(cls, builder: "ConfigBuilder[Self]") -> Self:
        """Create config from a builder."""
        inst = builder.build()
        if not isinstance(inst, cls):
            raise TypeError(f"ConfigBuilder.build() returned {type(inst)!r}, expected {cls!r}")
        return inst


class ConfigBuilder(Generic[C], ABC):
    """Generic base for staged config construction with validation.

    Concrete builders are dataclasses with optional fields.  They
    implement ``validate`` to collect errors into a ``ValidationError``
    and ``build`` to produce the final ``Config`` instance::

        @dataclass
        class MyConfigBuilder(ConfigBuilder[MyConfig]):
            foo: str | None = None
            bar: int | None = None

            def validate(self) -> None:
                errors: list[Exception] = []
                if self.foo is None:
                    errors.append(ValueError("foo is required"))
                if errors:
                    raise ValidationError(errors)

            def build(self) -> MyConfig:
                self.validate()
                return MyConfig(foo=self.foo, bar=self.bar)
    """

    # --- subclass hooks --- #

    @abstractmethod
    def build(self) -> C:
        """Build a config instance."""
        raise NotImplementedError

    @abstractmethod
    def validate(self) -> None:
        """Validate builder fields."""
        raise NotImplementedError

    # --- file io --- #

    @classmethod
    def from_file(
        cls,
        *files: Path,
        section: str | None = None,
        encoding: str | None = "utf-8",
        strict_type: bool = True,
    ) -> Self:
        """Build this config builder from one or more TOML/YAML files."""
        return dataclass_from_file(
            cls,
            *files,
            section=section,
            encoding=encoding,
            strict_type=strict_type,
        )

    def to_file(self, file: Path, *, encoding: str | None = "utf-8") -> None:
        """Dump this config builder to a TOML or YAML file."""
        dataclass_to_file(self, file, encoding=encoding)
