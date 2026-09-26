"""Root bases for physical modules and their cross-module data objects."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Self, dataclass_transform, final

import torch
import torch.nn as nn
from torch import Tensor

from .base_only_mixin import BaseOnlyMixin
from .dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin, map_single_tensor_fields
from .serialize_mixin import SerializeMixin
from .validate_mixin import ValidateMixin

DEFAULT_T__K: float = 300.0
"""Initial temperature of every physical module."""


@dataclass_transform(frozen_default=True, kw_only_default=True)
@dataclass(frozen=True, kw_only=True)
class ConfigBase(BaseOnlyMixin, SerializeMixin, ValidateMixin, base_only=True):
    """Base for immutable module configurations.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__` or `__post_init__`; this
    base supplies a frozen, keyword-only dataclass whose construction ends in
    `validate`.

    Extend `validate` for cross-field or physical-domain checks and call the
    parent implementation first. Direct construction runs those checks but does
    not coerce values according to annotations; typed loading through
    `from_dict` or `from_file` also checks field types. Inherited serialization
    exports config fields, not any module's fabricated or programmed state.
    """

    def __init_subclass__(cls, *, base_only: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(base_only=base_only, **kwargs)
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
class PolicyBase(BaseOnlyMixin, SerializeMixin, ValidateMixin, base_only=True):
    """Base for immutable module runtime policies.

    A subclass declares its fields as annotations without initial values, and
    must not apply `@dataclass` or define `__init__` or `__post_init__`; this
    base supplies a frozen, keyword-only dataclass whose construction ends in
    `validate`.
    """

    def __init_subclass__(cls, *, base_only: bool = False, **kwargs: object) -> None:
        super().__init_subclass__(base_only=base_only, **kwargs)
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


class SnapBase(TensorDataClassMixin, PyTreeDataClassMixin, BaseOnlyMixin, base_only=True):
    """Registered snapshot PyTree with tensor transforms under one per-call layout."""

    # === Public API ===

    @final
    def expand(self, shape: tuple[int, ...]) -> Self:
        """Apply `t.expand(shape)` to each tensor field."""
        return map_single_tensor_fields(lambda t: t.expand(shape), self)

    @final
    def flatten_axes(self, start_dim: int, end_dim: int) -> Self:
        """Apply `t.flatten(start_dim, end_dim)` to each tensor field."""
        return map_single_tensor_fields(lambda t: t.flatten(start_dim, end_dim), self)

    @final
    def index_select(self, dim: int, index: Tensor) -> Self:
        """Apply `t.index_select(dim, index)` to each tensor field."""
        return map_single_tensor_fields(lambda t: t.index_select(dim, index), self)


class DcopBase(TensorDataClassMixin, BaseOnlyMixin, base_only=True):
    """Immutable operating-point results for electrical solvers.

    Declare result fields with annotations and describe terminal signs, units,
    and layout beside each field. The inherited dataclass transform supplies a
    keyword-only constructor and identity equality. Do not decorate descendants
    with `dataclass` or define their constructor. Tensor storage remains
    mutable; callers should treat returned operating points as read-only
    observations.
    """


class ModuleBase(BaseOnlyMixin, nn.Module, base_only=True):
    """Base for physical modules with configuration and lifecycle hooks.

    Inherit exactly one accounting role: `ProfileModule` or `NonProfileModule`.
    Place registered sources before fabrication and programming. Sources move
    with the module but are excluded from `state_dict`; ordinary fabricated and
    programmed tensors require regeneration after placement changes.

    Fabrication visits each node before its NeuroX descendants, including those
    inside plain containers. Override `_sample_fabrication_variation` for local
    state only; programming is dispatched by its owner.

    Temperature starts at `DEFAULT_T__K` without invoking a hook. Updates visit
    parents before children and call `_on_temperature_changed`; existing physical
    realizations change only through explicit lifecycle operations.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
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
    @torch.no_grad()
    def stamp_names(self, qualified_name: str = "") -> None:
        """Assign paths to this node and its registered NeuroX descendants.

        Use the empty prefix for a standalone root. Child paths follow
        registered PyTorch module names, including intervening containers.
        Restamp after changing the tree, before collecting observations. For an
        assembled model with possible shared bindings, use `neurox.stamp_names`,
        which checks unique ownership first.

        Args:
            qualified_name: Root path prefix; the empty string names a
                standalone root.
        """
        self.__qualified_name = qualified_name
        for relative_name, child in neurox_children(self):
            child_name = relative_name if not qualified_name else f"{qualified_name}.{relative_name}"
            child.stamp_names(child_name)

    @final
    @torch.no_grad()
    def fabricate(self) -> None:
        """Replace this subtree's static manufacturing realization.

        Place tensor sources and set temperature first. The local fabrication
        hook runs before child hooks, including children reached through
        ordinary module containers. This call samples randomness for enabled
        static non-idealities; it does not program stored values. Program again
        when stored state depends on the newly fabricated parameters. Call
        outside compiled numerical work.
        """
        self._sample_fabrication_variation()
        for _, child in neurox_children(self):
            child.fabricate()

    @property
    @final
    def inst_count(self) -> int:
        """Physical instance count, the product of `inst_shape` (one for `()`)."""
        return math.prod(self.inst_shape)

    @property
    @final
    def qualified_name(self) -> str:
        """Hierarchical name assigned by `stamp_names`.

        Raises:
            RuntimeError: The module has not been stamped.
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
        """Rebuild this module's own static variation from nominal sources.

        Override when the model has fabricated state. The default does nothing.
        `fabricate` calls this hook before visiting descendants; do not recurse
        into children here or assume their new realization is already available.
        Use the current temperature and the placed sources' device and dtype.
        Repeated calls replace the local realization. Extend a parent hook with
        `super()` when the parent also owns fabricated state. Programming is a
        separate operation.
        """

    # === Tools for subclass and internal use ===

    @final
    def _register_nonpersistent_buffer(self, name: str, tensor: Tensor) -> None:
        """Register a tensor source for placement but not state export.

        Use during construction for nominal tensors needed by later fabrication
        or execution. PyTorch `.to()` moves the source, while `state_dict()`
        excludes it. This helper does not move ordinary fabricated or programmed
        attributes; recreate those states after changing device or computation
        dtype.

        Args:
            name: Name under which PyTorch registers the source buffer.
            tensor: Nominal tensor source that should follow module placement.
        """
        self.register_buffer(name, tensor, persistent=False)


class ProfileModule(ModuleBase, ABC, base_only=True):
    """Physical-module family with independently reported local costs.

    Families implement both per-instance static properties, either from
    configuration or from their own physical model. The public properties
    scale them by `inst_count` and exclude independently profiled children.
    Operations submit dynamic-energy contributions under their own path or a
    named child. Implementations account for each physical contribution once;
    collection does not change electrical behavior. No dynamic event is required
    for an operation with no modeled switching cost.

    Profiling retains the first `_profile_leading_rank` observation axes and
    sums contributions within each basic operation. The model owner configures
    this prefix so each retained element describes one basic operation of the
    owning unit. The rank starts at zero; batched operations require an explicit
    subtree setting before collection.

    Observations travel through the recorder's runtime submission. A CPU source
    identity stays on the host when the module moves devices, and names are
    resolved when observations arrive, so naming and collection size do not
    specialize numerical execution.
    """

    def __init__(
        self,
        *,
        config: ConfigBase,
        policy: PolicyBase,
        inst_shape: tuple[int, ...],
    ) -> None:
        from neurox.api.profiler import Profiler

        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._profile_leading_rank = 0
        self._profile_identity = Profiler.register_source(self)

    def __setstate__(self, state: dict[str, Any]) -> None:
        from neurox.api.profiler import Profiler

        super().__setstate__(state)
        self._profile_identity = Profiler.register_source(self)

    # === Public API ===

    @final
    def set_profile_leading_rank(self, leading_rank: int) -> None:
        """Set retained observation axes on this module and its profiled descendants.

        Call after assembling the model, before profiling or compiled execution.
        Keep the setting fixed throughout one measurement. Subtree updates
        overwrite earlier settings; newly attached children keep their own
        setting until the next update. Plain containers and non-profile modules
        pass the update through without acquiring profiling state.

        The model owner preserves all independent basic-operation positions in
        this prefix. A unit's children retain the same positions while summing
        that operation's internal contributions; axis scheduling is not inferred.
        """
        if leading_rank < 0:
            raise ValueError("profile leading rank must be non-negative")
        self._profile_leading_rank = leading_rank
        for _, child in neurox_profile_children(self):
            child.set_profile_leading_rank(leading_rank)

    @property
    @final
    def area__um2(self) -> float:
        """Local area of all instances, excluding independently profiled children."""
        return self._area_per_inst__um2 * self.inst_count

    @property
    @final
    def leakage__uW(self) -> float:
        """Local leakage of all instances, excluding independently profiled children."""
        return self._leakage_per_inst__uW * self.inst_count

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def _area_per_inst__um2(self) -> float:
        """Local area of one instance, excluding all child hardware."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _leakage_per_inst__uW(self) -> float:
        """Local leakage of one powered instance, excluding child hardware."""
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    @final
    def _is_profiler_active(self) -> bool:
        """Return whether this operation would submit to an active profiler.

        Use to avoid computing energy-only intermediates outside collection. The
        branch must not change the operation's numerical outputs or physical
        state.

        Returns:
            True if a profiler is currently active; otherwise False.
        """
        from neurox.api.profiler import Profiler

        return Profiler.active()

    @final
    def _record_dynamic_energy(self, dynamic_energy__fJ: Tensor, *, channel: str | None = None) -> None:
        """Submit energy when profiling, retaining the configured observation prefix.

        Compute energy under `torch.no_grad()` or an equivalent guard. Reassemble
        chunks before submission and include each physical instance and access
        once; trailing axes are summed. Scalar constants may be expanded views.

        Args:
            dynamic_energy__fJ: Energy over full caller extents and billed
                trailing axes. Singleton dimensions do not request broadcasting.
                Shape: `[*caller_leading, ...]`.
            channel: Optional billing segment; a matching child shares the
                same measurement item.
        """
        from neurox.api.profiler import Profiler

        profiler = Profiler.current()
        if profiler is None:
            return

        rank = self._profile_leading_rank
        if rank > dynamic_energy__fJ.ndim:
            raise ValueError("profile leading rank exceeds the submitted tensor rank")
        if rank < dynamic_energy__fJ.ndim:
            # Shape: [*observation, *work] -> [*observation]
            dynamic_energy__fJ = dynamic_energy__fJ.sum(dim=tuple(range(rank, dynamic_energy__fJ.ndim)))

        profiler.submit_dynamic_energy(dynamic_energy__fJ, source=self._profile_identity, channel=channel)

    @final
    def _record_latency(self, latency__ns: Tensor) -> None:
        """Submit already computed durations with the retained sample layout.

        Outside a profiling context nothing is submitted. No axes are reduced;
        a constant basic-operation duration may be expanded over sample positions.

        Args:
            latency__ns: Durations aligned with the module's energy positions.
                Shape: `[*measurement]`.
        """
        from neurox.api.profiler import Profiler

        profiler = Profiler.current()
        if profiler is None:
            return

        profiler.submit_latency(latency__ns, source=self._profile_identity)


class NonProfileModule(ModuleBase, base_only=True):
    """Physical-module family without independent PPA reporting.

    Its physical costs are accounted for by an owner or outside the model's
    scope. It exposes no static-cost or energy-emission interface. Profiled
    children still report their own costs when held below this node.
    """


def neurox_children(module: nn.Module) -> Iterator[tuple[str, ModuleBase]]:
    """Yield the nearest NeuroX descendants.

    Plain module containers are traversed; descent stops at each NeuroX module.
    Repeated references are deduplicated by identity, retaining the first path
    in registered-child order.

    Yields:
        Pairs of relative registered paths and NeuroX modules.
    """
    seen: set[ModuleBase] = set()
    for name, child in module.named_children():
        if isinstance(child, ModuleBase):
            candidates = iter(((name, child),))
        else:
            candidates = (
                (f"{name}.{relative_name}", descendant) for relative_name, descendant in neurox_children(child)
            )
        for relative_name, descendant in candidates:
            if descendant in seen:
                continue
            seen.add(descendant)
            yield relative_name, descendant


def neurox_roots(model: nn.Module) -> Iterator[tuple[str, ModuleBase]]:
    """Yield the outermost NeuroX modules with their paths from `model`.

    Plain containers are traversed; descent stops at each NeuroX module.
    Repeated references retain their first path in registered-child order.

    Yields:
        Relative paths and outermost NeuroX modules; the path is empty when
        `model` itself is a NeuroX module.
    """
    if isinstance(model, ModuleBase):
        yield "", model
        return
    yield from neurox_children(model)


def neurox_named_modules(model: nn.Module) -> Iterator[tuple[str, ModuleBase]]:
    """Visit every NeuroX node with its path from `model`, including the root.

    Plain containers are traversed. Repeated bindings appear at every path so
    ownership checks can detect them before a caller assigns physical costs.

    Yields:
        Relative paths and NeuroX modules in tree traversal order.
    """
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, ModuleBase):
            yield name, module


def neurox_profile_children(module: nn.Module) -> Iterator[tuple[str, ProfileModule]]:
    """Yield the nearest profile descendants, excluding the supplied module.

    Plain containers and non-profile modules are traversed; descent stops at
    each profile module. Repeated references retain their first relative path
    in registered-child order.

    Yields:
        Pairs of relative registered paths and profile modules.
    """
    seen: set[ProfileModule] = set()
    for name, child in module.named_children():
        if isinstance(child, ProfileModule):
            candidates = iter(((name, child),))
        else:
            candidates = (
                (f"{name}.{relative_name}", descendant) for relative_name, descendant in neurox_profile_children(child)
            )
        for relative_name, descendant in candidates:
            if descendant in seen:
                continue
            seen.add(descendant)
            yield relative_name, descendant


def neurox_profile_roots(model: nn.Module) -> Iterator[tuple[str, ProfileModule]]:
    """Yield the outermost profile modules, crossing non-profile owners.

    A profile root covers its own descendants. Paths remain relative to the
    supplied model, including ordinary containers and non-profile ancestors.
    """
    if isinstance(model, ProfileModule):
        yield "", model
        return
    yield from neurox_profile_children(model)


def neurox_profile_modules(model: nn.Module) -> Iterator[tuple[str, ProfileModule]]:
    """Yield independently profiled modules at every depth of the model tree.

    Yields:
        Relative paths and physical modules that own local PPA costs.
    """
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, ProfileModule):
            yield name, module
