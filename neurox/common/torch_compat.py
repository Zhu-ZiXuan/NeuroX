"""Typed and structured boundaries around PyTorch-specific interfaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, overload

import torch
from torch import Tensor
from torch._higher_order_ops.map import map as _map
from torch._higher_order_ops.scan import scan
from torch.utils import _pytree as pytree

__all__ = [
    "torch_assert_async",
    "torch_compiler_disable",
    "torch_cond",
    "torch_map",
    "torch_scan",
    "torch_while_loop",
]


class _Decorator(Protocol):
    def __call__[**P, T](self, fn: Callable[P, T]) -> Callable[P, T]: ...


class _CompilerDisable(Protocol):
    @overload
    def __call__[**P, T](
        self, fn: Callable[P, T], recursive: bool = True, *, reason: str | None = None
    ) -> Callable[P, T]: ...

    @overload
    def __call__(self, fn: None = None, recursive: bool = True, *, reason: str | None = None) -> _Decorator: ...


torch_compiler_disable: _CompilerDisable = torch.compiler.disable


def torch_assert_async(condition: Tensor, message: str) -> None:
    """Wrap ``torch._assert_async(input: Tensor, assert_msg: str)``."""
    torch._assert_async(condition, message)  # noqa: SLF001


def torch_cond[InputsT, OutputT](
    pred: Tensor | bool,
    true_fn: Callable[[InputsT], OutputT],
    false_fn: Callable[[InputsT], OutputT],
    inputs: InputsT,
    *,
    output_template: OutputT,
) -> OutputT:
    """Adapt `torch.cond` to one structured input and a structured output.

    Args:
        output_template: Result PyTree structure; tensor values and metadata
            are unused.
    """
    # --- 1. Flatten pytree and check params ---

    def _flatten_pytree(tree: object, *, name: str) -> tuple[tuple[Tensor, ...], pytree.TreeSpec]:
        leaves, spec = pytree.tree_flatten(tree)
        if not all(isinstance(leaf, Tensor) for leaf in leaves):
            raise TypeError(f"{name} must contain only Tensor leaves")
        return tuple(leaves), spec

    flat_inputs, input_spec = _flatten_pytree(inputs, name="inputs")
    _, output_spec = _flatten_pytree(output_template, name="output_template")

    # --- 2. Define unflatten functions ---

    def _unflatten_inputs(flat_inputs: tuple[Tensor, ...]) -> InputsT:
        return cast(InputsT, pytree.tree_unflatten(flat_inputs, input_spec))

    def _unflatten_output(flat_output: tuple[Tensor, ...]) -> OutputT:
        return cast(OutputT, pytree.tree_unflatten(flat_output, output_spec))

    # --- 3. Define wrapper functions ---

    def flatten_branch(fn: Callable[[InputsT], OutputT], flat_inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        output = fn(_unflatten_inputs(flat_inputs))
        flat_output, branch_spec = _flatten_pytree(output, name="branch output")
        if branch_spec != output_spec:
            raise TypeError("branch output must match the output_template PyTree structure")
        return flat_output

    def flat_true_fn(*flat_inputs: Tensor) -> tuple[Tensor, ...]:
        return flatten_branch(true_fn, flat_inputs)

    def flat_false_fn(*flat_inputs: Tensor) -> tuple[Tensor, ...]:
        return flatten_branch(false_fn, flat_inputs)

    # --- 4. Run pytorch cond ---

    flat_output = torch.cond(pred, flat_true_fn, flat_false_fn, flat_inputs)

    # --- 5. Unflatten pytree ---

    return _unflatten_output(flat_output)


type _WhileLoopLeaf = Tensor | int | torch.SymInt


def torch_while_loop[CarryT](
    cond_fn: Callable[[CarryT], Tensor | bool],
    body_fn: Callable[[CarryT], CarryT],
    carried_state: CarryT,
) -> CarryT:
    """Adapt `torch.while_loop` to callbacks receiving one structured state."""
    # --- 1. Flatten pytree and check params ---

    def _flatten_pytree(tree: object, *, name: str) -> tuple[tuple[_WhileLoopLeaf, ...], pytree.TreeSpec]:
        leaves, spec = pytree.tree_flatten(tree)
        if not all(isinstance(leaf, (Tensor, int, torch.SymInt)) for leaf in leaves):
            raise TypeError(f"{name} must contain only Tensor, int, or SymInt leaves")
        return tuple(leaves), spec

    flat_carry, carry_spec = _flatten_pytree(carried_state, name="carried_state")

    # --- 2. Define unflatten functions ---

    def _unflatten_carry(flat_carry: tuple[_WhileLoopLeaf, ...]) -> CarryT:
        return cast(CarryT, pytree.tree_unflatten(flat_carry, carry_spec))

    # --- 3. Define wrapper functions ---

    def flat_cond_fn(*flat_carry: _WhileLoopLeaf) -> Tensor | bool:
        return cond_fn(_unflatten_carry(flat_carry))

    def flat_body_fn(*flat_carry: _WhileLoopLeaf) -> tuple[_WhileLoopLeaf, ...]:
        new_carry = body_fn(_unflatten_carry(flat_carry))
        new_carry_leaves, new_carry_spec = _flatten_pytree(new_carry, name="body output")
        if new_carry_spec != carry_spec:
            raise TypeError(
                "body output must preserve the carried-state PyTree structure; "
                f"expected {carry_spec}, got {new_carry_spec}"
            )
        return new_carry_leaves

    # --- 4. Run pytorch while loop ---

    flat_carry = torch.while_loop(flat_cond_fn, flat_body_fn, flat_carry)

    # --- 5. Unflatten pytree ---

    return _unflatten_carry(flat_carry)


def torch_scan[CarryT, InputT, OutputT](
    combine_fn: Callable[[CarryT, InputT], tuple[CarryT, OutputT]],
    init: CarryT,
    xs: InputT,
    *,
    dim: int = 0,
    reverse: bool = False,
    output_template: OutputT,
) -> tuple[CarryT, OutputT]:
    """Adapt PyTorch's higher-order scan to structured carry, inputs, and outputs.

    Args:
        output_template: Output PyTree structure, including optional fields;
            tensor values and metadata are unused.
        dim: Negative values are resolved against the first input leaf.

    Returns:
        Final carry and outputs in input order, including reverse traversal.
        Each output's iteration axis occupies the resolved `dim` if that axis
        exists in the stacked tensor, otherwise axis zero.
    """
    # --- 1. Flatten pytree and check params ---

    def _flatten_pytree(tree: object, *, name: str) -> tuple[tuple[Tensor, ...], pytree.TreeSpec]:
        leaves, spec = pytree.tree_flatten(tree)
        if not all(isinstance(leaf, Tensor) for leaf in leaves):
            raise TypeError(f"{name} must contain only Tensor leaves")
        return tuple(leaves), spec

    flat_carry, carry_spec = _flatten_pytree(init, name="init")
    flat_xs, xs_spec = _flatten_pytree(xs, name="xs")
    _, output_spec = _flatten_pytree(output_template, name="output_template")
    if not flat_carry or not flat_xs:
        raise ValueError("init and xs must each contain at least one Tensor leaf")

    scan_dim = dim if dim >= 0 else dim + flat_xs[0].ndim
    if scan_dim < 0 or any(tensor.ndim <= scan_dim for tensor in flat_xs):
        raise ValueError(f"all xs leaves must have dimension {dim}")

    scan_length = flat_xs[0].shape[scan_dim]
    if scan_length == 0:
        raise ValueError("the scan dimension must have positive length")
    if any(tensor.shape[scan_dim] != scan_length for tensor in flat_xs[1:]):
        raise ValueError("all xs leaves must have the same scan dimension length")

    flat_xs = tuple(tensor.movedim(scan_dim, 0) for tensor in flat_xs)
    if reverse:
        flat_xs = tuple(tensor.flip(0) for tensor in flat_xs)

    # --- 2. Define unflatten functions ---

    def _unflatten_carry(flat_carry: tuple[Tensor, ...]) -> CarryT:
        return cast(CarryT, pytree.tree_unflatten(flat_carry, carry_spec))

    def _unflatten_input(flat_input: tuple[Tensor, ...]) -> InputT:
        return cast(InputT, pytree.tree_unflatten(flat_input, xs_spec))

    def _unflatten_output(flat_output: tuple[Tensor, ...]) -> OutputT:
        return cast(OutputT, pytree.tree_unflatten(flat_output, output_spec))

    # --- 3. Define wrapper functions ---

    def flat_combine_fn(
        flat_carry: tuple[Tensor, ...],
        flat_input: tuple[Tensor, ...],
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        new_carry, new_output = combine_fn(_unflatten_carry(flat_carry), _unflatten_input(flat_input))
        new_flat_carry, new_carry_spec = _flatten_pytree(new_carry, name="carry output")
        new_flat_output, new_output_spec = _flatten_pytree(new_output, name="scan output")
        if new_carry_spec != carry_spec:
            raise TypeError("carry output must preserve the init PyTree structure")
        if new_output_spec != output_spec:
            raise TypeError("scan output must match the output_template PyTree structure")
        return new_flat_carry, new_flat_output

    # --- 4. Run pytorch scan ---

    flat_carry, stacked_flat_output = scan(flat_combine_fn, flat_carry, flat_xs, dim=0, reverse=False)

    # --- 5. Unflatten pytree ---

    if reverse:
        stacked_flat_output = tuple(tensor.flip(0) for tensor in stacked_flat_output)
    stacked_flat_output = tuple(
        tensor.movedim(0, scan_dim) if scan_dim < tensor.ndim else tensor for tensor in stacked_flat_output
    )

    return _unflatten_carry(flat_carry), _unflatten_output(stacked_flat_output)


def torch_map[InputT, OutputT](
    fn: Callable[[InputT], OutputT],
    xs: InputT,
    *,
    output_template: OutputT,
) -> OutputT:
    """Map leading-axis input slices to structured outputs independently.

    `xs` must be a nonempty Tensor PyTree whose leaves have the same positive
    size on axis 0. Each call receives one slice; invariant values are captured
    by `fn`. Registered dataclasses are supported as both inputs and outputs,
    including optional fields recorded as absent by PyTree registration.

    The body must be capturable by `torch.compile`, with no input mutation
    or output aliasing. Input and output PyTree leaves must be Tensors.

    Args:
        output_template: Output PyTree structure, including optional fields;
            tensor values and metadata are unused.

    Returns:
        Outputs stacked on axis 0, with the template's PyTree structure.
    """
    # --- 1. Flatten pytree and check params ---

    def _flatten_pytree(tree: object, *, name: str) -> tuple[tuple[Tensor, ...], pytree.TreeSpec]:
        leaves, spec = pytree.tree_flatten(tree)
        if not all(isinstance(leaf, Tensor) for leaf in leaves):
            raise TypeError(f"{name} must contain only Tensor leaves")
        return tuple(leaves), spec

    flat_xs, xs_spec = _flatten_pytree(xs, name="xs")
    _, output_spec = _flatten_pytree(output_template, name="output_template")
    if not flat_xs:
        raise ValueError("xs must contain at least one Tensor leaf")
    if any(leaf.ndim == 0 for leaf in flat_xs):
        raise ValueError("all xs Tensor leaves must have a leading dimension")

    # --- 2. Define unflatten functions ---

    def _unflatten_input(flat_input: tuple[Tensor, ...]) -> InputT:
        return cast(InputT, pytree.tree_unflatten(flat_input, xs_spec))

    def _unflatten_output(flat_output: tuple[Tensor, ...]) -> OutputT:
        return cast(OutputT, pytree.tree_unflatten(flat_output, output_spec))

    # --- 3. Define wrapper function ---

    def flat_fn(flat_input: tuple[Tensor, ...], *_: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        output = fn(_unflatten_input(flat_input))
        flat_output, new_output_spec = _flatten_pytree(output, name="map output")
        if new_output_spec != output_spec:
            raise TypeError("map output must match the output_template PyTree structure")
        return flat_output

    # --- 4. Run pytorch map ---

    flat_output = _map(flat_fn, flat_xs)

    # --- 5. Unflatten pytree ---

    return _unflatten_output(flat_output)
