"""Root bases for physical modules and their cross-module data objects."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, dataclass_transform, final

import torch
import torch.nn as nn
from torch import Tensor

from neurox.api.profiler import Profiler

from .dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin, map_single_tensor_fields
from .serialize_mixin import SerializeMixin
from .validate_mixin import ValidateMixin

DEFAULT_T__K: float = 300.0
"""Initial temperature of every physical module."""


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class ConfigBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module configurations.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__` or `__post_init__`; this
    base supplies a frozen, keyword-only dataclass whose construction ends in
    `validate`.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    # === For subclass to implement or override ===

    def validate(self) -> None:
        """Check the local constraints on this configuration.

        Overrides call `super().validate()` before checking their own fields
        and raise `ValueError` when a constraint is violated.
        """


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class PolicyBase(SerializeMixin, ValidateMixin, ABC):
    """Base for immutable module runtime policies.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__` or `__post_init__`; this
    base supplies a frozen, keyword-only dataclass whose construction ends in
    `validate`.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__init__" in cls.__dict__:
            raise TypeError(f"{cls.__qualname__} must declare dataclass fields, not __init__()")
        for name in cls.__annotations__:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name} carries an initial value; declare the annotation alone")
        dataclass(frozen=True, kw_only=True)(cls)

    @final
    def __post_init__(self) -> None:
        self.validate()

    # === For subclass to implement or override ===

    def validate(self) -> None:
        """Check the local constraints on this runtime policy.

        Overrides call `super().validate()` before checking their own fields
        and raise `ValueError` when a constraint is violated.
        """


class SnapBase(TensorDataClassMixin, PyTreeDataClassMixin):
    """Registered snapshot PyTree with tensor transforms under one per-call layout."""

    # === Public API ===

    @final
    def expand(self, shape: tuple[int, ...]) -> Self:
        """Apply ``t.expand(shape)`` on each tensor field."""
        return map_single_tensor_fields(lambda t: t.expand(shape), self)

    @final
    def flatten_axes(self, start_dim: int, end_dim: int) -> Self:
        """Apply ``t.flatten(start_dim, end_dim)`` on each tensor field."""
        return map_single_tensor_fields(lambda t: t.flatten(start_dim, end_dim), self)

    @final
    def index_select(self, dim: int, index: Tensor) -> Self:
        """Apply ``t.index_select(dim, index)`` on each tensor field."""
        return map_single_tensor_fields(lambda t: t.index_select(dim, index), self)


class DcopBase(TensorDataClassMixin):
    pass


class ModuleBase(nn.Module, ABC):
    """Base for config- and policy-managed physical modules.

    Construct the owned module tree, place it on its device, then fabricate and
    program before execution. Construction registers tensor sources with
    `_register_nonpersistent_buffer`. These buffers migrate with the module and
    are excluded from `state_dict`.

    Fabricated and programmed tensors are ordinary attributes produced by their
    lifecycle operations. They are neither migrated nor serialized with the
    module. Moving a module after materializing them requires fabrication and
    programming to run again. Per-call snapshots are local results.

    `fabricate()` invokes `_sample_fabrication_variation()` on this node before
    its registered NeuroX descendants, including those inside plain containers.
    Each hook rebuilds only its own state from nominal sources; it does not
    traverse children. Programming is dispatched explicitly by the owner.

    Temperature starts at `DEFAULT_T__K`. Construction stores it without
    invoking `_on_temperature_changed`. Temperature-dependent computations
    read `T__K` at their lifecycle or execution point. Temperature updates
    visit this node before its NeuroX descendants, including plain containers.

    Stamp names after assembling the tree and before profiling. Re-stamping
    updates the subtree names; rebuilding from configuration requires a new
    fabrication and programming lifecycle.

    A family inherits either `ProfileModule` or `NonProfileModule`; its
    implementations retain that accounting identity, including ideal models.
    An instance with neither or both identities is rejected. Both branches
    participate in the same naming, fabrication, and temperature walks.

    Args:
        inst_shape: Hardware-instance shape. Physical axes encode multiplicity;
            singleton axes may reserve positions for runtime broadcasting.
    """

    __qualified_name: str

    def __init__(
        self,
        *,
        config: ConfigBase,
        policy: PolicyBase,
        inst_shape: tuple[int, ...],
    ) -> None:
        if isinstance(self, ProfileModule) == isinstance(self, NonProfileModule):
            raise TypeError(f"{type(self).__qualname__} must inherit exactly one of ProfileModule and NonProfileModule")
        nn.Module.__init__(self)
        if any(size <= 0 for size in inst_shape):
            raise ValueError(f"inst_shape extents must be positive; got {inst_shape}")
        self.config = config
        self.policy = policy
        self.inst_shape = inst_shape
        self._T__K = DEFAULT_T__K

    # === Public API ===

    @property
    @final
    def T__K(self) -> float:
        """Current temperature, changed through `set_temperature`."""
        return self._T__K

    @final
    @torch.no_grad()
    def set_temperature(self, T__K: float) -> None:
        """Update temperature and its direct dependencies across this subtree.

        Call between executions, outside compiled computations. Fabricated
        and programmed state remains unchanged until its next explicit
        lifecycle call. Newly attached children retain their own temperature
        until the next subtree update.

        Raises:
            ValueError: Temperature is not finite and strictly positive, or
                a subclass cannot represent a derived parameter.
        """
        if not (math.isfinite(T__K) and T__K > 0.0):
            raise ValueError(f"require: T__K ({T__K}) finite and > 0.0")
        self._T__K = T__K
        self._on_temperature_changed()
        for _, child in neurox_children(self):
            child.set_temperature(T__K)

    @final
    def stamp_names(self, *, qualified_name: str = "") -> None:
        """Stamp this module and its NeuroX subtree with hierarchical names."""
        self.__qualified_name = qualified_name
        for relative_name, child in neurox_children(self):
            child_name = relative_name if not qualified_name else f"{qualified_name}.{relative_name}"
            child.stamp_names(qualified_name=child_name)

    @final
    @torch.no_grad()
    def fabricate(self) -> None:
        """Resample static manufacturing variation across this module subtree."""
        self._sample_fabrication_variation()
        for _, child in neurox_children(self):
            child.fabricate()

    @property
    @final
    def inst_count(self) -> int:
        return math.prod(self.inst_shape)

    @property
    @final
    def qualified_name(self) -> str:
        """Hierarchical name the module's tree stamped onto it.

        Raises:
            RuntimeError: No tree has stamped this module yet.
        """
        try:
            return self.__qualified_name
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__} carries no name stamp; "
                "call neurox.stamp_names(model) once the model is assembled"
            ) from None

    # === For subclass to implement or override ===

    def _on_temperature_changed(self) -> None:
        """Refresh this node's direct dependencies from its current `T__K`.

        The default does nothing. Overrides update local scalar parameters
        and registered sources while preserving buffer device and dtype.
        They do not traverse children, sample randomness, or change fabricated
        or programmed state. Overrides extending a parent's update call it
        before updating their own dependencies.
        """

    def _sample_fabrication_variation(self) -> None:
        pass

    # === Tools for subclass and internal use ===

    @final
    def _register_nonpersistent_buffer(self, name: str, tensor: Tensor) -> None:
        nn.Module.register_buffer(self, name, tensor, persistent=False)


class ProfileModule(ModuleBase):
    """Physical-module family with independently reported local costs.

    Families implement both per-instance static properties, either from
    configuration or from their own physical model. The public properties
    scale them by `inst_count` and exclude independently profiled children.
    Operations submit only the dynamic energy this module owns; collection
    does not change electrical behavior. No dynamic event is required for an
    operation with no modeled switching cost.
    """

    # === Public API ===

    @property
    @final
    def area__um2(self) -> float:
        return self._area_per_inst__um2 * self.inst_count

    @property
    @final
    def leakage__uW(self) -> float:
        return self._leakage_per_inst__uW * self.inst_count

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def _area_per_inst__um2(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def _leakage_per_inst__uW(self) -> float:
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _is_profiler_active(self) -> bool:
        return Profiler.active()

    @final
    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Submit this call's local dynamic energy to the active profiler.

        Outside a profiling context nothing is submitted. Compute billed
        energy under `torch.no_grad()` or an equivalent guard.

        Preserve the caller's leading axes at their full extents; reassemble
        chunked results before submitting them. The profiler retains those
        axes and sums all trailing axes, so include each billed physical
        instance and access exactly once. Constant energy may be expanded
        from a scalar without materializing the billed layout.

        Args:
            dynamic_energy__fJ: Energy over the caller's full leading extents
                and billed trailing axes. Expanded views are supported;
                singleton dimensions do not request implicit broadcasting.
                Shape: `[*caller_leading, ...]`.
            channel: Optional virtual submodule to bill under.

        Raises:
            RuntimeError: The emitter carries no name stamp while a profiler
                is collecting.
        """
        ledger = Profiler.current()
        if ledger is None:
            return
        Profiler.submit(
            ledger.lay_out(
                qualified_name=self.qualified_name,
                dynamic_energy__fJ=dynamic_energy__fJ,
                channel=channel,
            )
        )


class NonProfileModule(ModuleBase):
    """Physical-module family without independent PPA reporting.

    Its physical costs are accounted for by an owner or outside the model's
    scope. It exposes no static-cost or energy-emission interface. Profiled
    children still report their own costs when held below this node.
    """


def neurox_roots(model: nn.Module) -> list[ModuleBase]:
    """Collect the outermost NeuroX modules `model` holds.

    The walk stops descending at the first `ModuleBase` it meets, so a root
    covers its own NeuroX children instead of listing them beside it. A plain
    container may hold several roots. Roots are deduplicated by identity and
    retain their first-appearance order.

    Returns:
        The outermost NeuroX modules.
    """
    if isinstance(model, ModuleBase):
        return [model]
    return [module for _, module in neurox_children(model)]


def neurox_children(
    module: nn.Module,
) -> list[tuple[str, ModuleBase]]:
    """Collect the nearest NeuroX descendants.

    Plain module containers are traversed; descent stops at each NeuroX module.
    Repeated references are deduplicated by identity, retaining the first path
    in registered-child order.

    Returns:
        Pairs of relative registered paths and NeuroX modules.
    """
    children: list[tuple[str, ModuleBase]] = []
    seen: set[ModuleBase] = set()
    for name, child in module.named_children():
        if isinstance(child, ModuleBase):
            candidates = [(name, child)]
        else:
            candidates = [
                (f"{name}.{relative_name}", descendant) for relative_name, descendant in neurox_children(child)
            ]
        for relative_name, descendant in candidates:
            if descendant in seen:
                continue
            seen.add(descendant)
            children.append((relative_name, descendant))
    return children
