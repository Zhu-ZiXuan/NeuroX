"""Compile scoped record outputs and submit them at the execution boundary.

See Also:
    docs/system_design/record_collection.md
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, overload

import torch
from torch import Tensor
from torch._ops import HigherOrderOperator
from torch.fx import GraphModule, Node
from torch.fx.node import Argument, map_arg
from torch.utils import _pytree as pytree

from neurox.common.recording import Submission, submission_for


@dataclass
class _Emission:
    submission: Submission
    arguments: dict[str, Any]


@dataclass
class _Scope:
    begin: Node
    kind: str
    shape: tuple[int, ...]
    axis: int
    emissions: list[_Emission] = field(default_factory=list)
    groups: list[list[_Emission]] = field(default_factory=list)
    matched: bool = False


def _attribute(module: GraphModule, path: str) -> torch.nn.Module:
    result = module
    for name in path.split("."):
        result = getattr(result, name)
    return result


def _place_axis(value: Tensor, axis: int) -> Tensor:
    destination = axis if axis < value.ndim else 0
    return value.movedim(0, destination)


def _flatten_group(value: Tensor, count: int, size: int) -> Tensor:
    if count == 1:
        if value.ndim == 0 or value.shape[0] != size:
            raise ValueError("Chunk records must retain their positional prefix")
        return value
    if value.ndim < 2 or tuple(value.shape[:2]) != (count, size):
        raise ValueError("Mapped chunk records must retain chunk and position axes")
    return value.flatten(0, 1)


def _restore_positions(value: Tensor, shape: tuple[int, ...]) -> Tensor:
    return value.reshape((*shape, *value.shape[1:]))


def _reduce_observations(value: Tensor, rank: int) -> Tensor:
    if not 0 <= rank <= value.ndim:
        raise ValueError("profile leading rank exceeds the restored record rank")
    if rank < value.ndim:
        return value.sum(dim=tuple(range(rank, value.ndim)))
    return value


def _payload_nodes(emission: _Emission) -> list[Node]:
    nodes: list[Node] = []

    def collect(node: Node) -> Node:
        nodes.append(node)
        return node

    for name in emission.submission.payloads:
        map_arg(emission.arguments[name], collect)
    return nodes


def _map_payload(emission: _Emission, transform: Callable[[Node], Node]) -> _Emission:
    arguments = dict(emission.arguments)
    for name in emission.submission.payloads:
        arguments[name] = map_arg(arguments[name], transform)
    return _Emission(emission.submission, arguments)


def _metadata(emission: _Emission) -> dict[str, Any]:
    return {key: value for key, value in emission.arguments.items() if key not in emission.submission.payloads}


def _contains_records(module: GraphModule) -> bool:
    return any(
        submission_for(node.target) is not None
        for child in module.modules()
        if isinstance(child, GraphModule)
        for node in child.graph.nodes
        if node.op == "call_function"
    )


def _remove_empty_scopes(module: GraphModule) -> None:
    for child in module.modules():
        if isinstance(child, GraphModule):
            for node in reversed(list(child.graph.nodes)):
                if node.op == "call_function" and node.target in (
                    torch.ops.neurox.record_scope_begin.default,
                    torch.ops.neurox.record_scope_end.default,
                ):
                    child.graph.erase_node(node)
            child.recompile()


def _transform(module: GraphModule, *, loop_body: bool) -> list[_Emission]:
    graph = module.graph
    emissions: list[_Emission] = []
    scopes: list[_Scope] = []

    def append(records: list[_Emission]) -> None:
        target = scopes[-1].emissions if scopes else emissions
        target.extend(records)

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target == torch.ops.neurox.record_scope_begin.default:
            kind, shape, axis = node.args
            scopes.append(_Scope(node, kind, tuple(shape), axis))
            continue
        if node.target == torch.ops.neurox.record_scope_end.default:
            if not scopes or scopes[-1].begin is not node.args[0]:
                raise ValueError("Record scopes must close in nesting order")
            scope = scopes.pop()
            with graph.inserting_before(node):
                records = _finish_scope(graph, scope)
            if scope.kind == "group" and scopes and scopes[-1].kind == "chunk":
                scopes[-1].groups.append(records)
            else:
                append(records)
            graph.erase_node(node)
            graph.erase_node(scope.begin)
            continue

        submission = submission_for(node.target)
        if submission is not None:
            if loop_body and not submission.allow_loop:
                raise ValueError("This record cannot be collected inside scan/map")
            arguments = dict(submission.signature.bind(*node.args, **node.kwargs).arguments)
            for name, parameter in submission.signature.parameters.items():
                if name not in arguments:
                    arguments[name] = parameter.default
            emission = _Emission(submission, arguments)
            # Preserve the observed value even if the caller mutates an alias later.
            with graph.inserting_before(node):
                emission = _map_payload(emission, lambda value: graph.call_function(torch.clone, (value,)))
            append([emission])
            if node.users:
                raise ValueError("A submission must not have functional outputs")
            graph.erase_node(node)
            continue

        kind = None
        if node.target == torch.ops.higher_order.scan:
            kind = "scan"
        elif node.target == torch.ops.higher_order.map_impl:
            kind = "map"
        owned = kind is not None and scopes and scopes[-1].kind == kind and not scopes[-1].matched
        if owned:
            scope = scopes[-1]
            scope.matched = True
            append(_lift_loop(module, node, kind=kind))
        elif isinstance(node.target, HigherOrderOperator):
            for argument in node.args:
                if isinstance(argument, Node) and argument.op == "get_attr" and isinstance(argument.target, str):
                    child = _attribute(module, argument.target)
                    if isinstance(child, GraphModule):
                        if _contains_records(child):
                            raise ValueError(
                                "Hidden records may cross only NeuroX run_scan/run_map scopes; "
                                f"unsupported boundary: {node.target}"
                            )
                        _remove_empty_scopes(child)

    if scopes:
        raise ValueError("Unclosed record scope")
    if loop_body and emissions:
        output = next(node for node in graph.nodes if node.op == "output")
        extra = [value for emission in emissions for value in _payload_nodes(emission)]
        output.args = (tuple(output.args[0]) + tuple(extra),)
    graph.lint()
    module.recompile()
    return emissions


def _loop_arguments(value: Argument) -> Sequence[Argument]:
    if not isinstance(value, (tuple, list)):
        raise TypeError("A captured loop requires flat argument sequences")
    return value


def _lift_loop(module: GraphModule, node: Node, *, kind: str) -> list[_Emission]:
    graph = module.graph
    reference = node.args[0]
    if not isinstance(reference, Node) or reference.op != "get_attr" or not isinstance(reference.target, str):
        raise TypeError("A collected loop must reference a captured subgraph")
    child = _attribute(module, reference.target)
    if not isinstance(child, GraphModule):
        raise TypeError("A collected loop must have a captured GraphModule body")
    output = next(value for value in child.graph.nodes if value.op == "output")
    output_count = len(output.args[0])
    records = _transform(child, loop_body=True)
    if not records:
        return []
    if kind == "scan":
        carry, xs, lifted = node.args[1:]
        invariant_start = len(_loop_arguments(carry)) + len(_loop_arguments(xs))
    else:
        xs, lifted = node.args[1:]
        invariant_start = len(_loop_arguments(xs))
    placeholders = [value for value in child.graph.nodes if value.op == "placeholder"]
    invariant_map: dict[Node, Argument] = dict(
        zip(placeholders[invariant_start:], _loop_arguments(lifted), strict=True)
    )

    def invariant(value: Node) -> Argument:
        if value not in invariant_map:
            raise ValueError("Record identities and metadata must be invariant across a loop")
        return invariant_map[value]

    cursor = node
    result: list[_Emission] = []
    for emission in records:
        arguments = dict(map_arg(_metadata(emission), invariant))
        for name in emission.submission.payloads:

            def lift(value: Node) -> Node:
                nonlocal cursor, output_count
                with graph.inserting_after(cursor):
                    value = graph.call_function(operator.getitem, (node, output_count))
                cursor = value
                output_count += 1
                return value

            arguments[name] = map_arg(emission.arguments[name], lift)
        result.append(_Emission(emission.submission, arguments))
    node.meta.pop("example_value", None)
    return result


def _finish_scope(graph: torch.fx.Graph, scope: _Scope) -> list[_Emission]:
    if scope.kind in ("scan", "map"):
        if not scope.matched:
            raise ValueError("A record loop scope must own one matching higher-order operator")
        records = scope.emissions
        if scope.kind == "scan" and scope.shape[0]:
            records = [
                _map_payload(record, lambda value: graph.call_function(torch.flip, (value, [0]))) for record in records
            ]
        return [
            _map_payload(record, lambda value: graph.call_function(_place_axis, (value, scope.axis)))
            for record in records
        ]
    if scope.kind == "group":
        count, size = scope.shape
        return [
            _map_payload(record, lambda value: graph.call_function(_flatten_group, (value, count, size)))
            for record in scope.emissions
        ]
    if scope.kind == "energy_sum":
        return [
            _map_payload(record, lambda value: graph.call_function(torch.sum, (value,), {"dim": scope.shape}))
            if record.submission.additive
            else record
            for record in scope.emissions
        ]
    if scope.kind == "energy_access":
        merged: list[_Emission] = []
        for emission in scope.emissions:
            previous = next(
                (
                    candidate
                    for candidate in merged
                    if emission.submission.additive
                    and candidate.submission == emission.submission
                    and _metadata(candidate) == _metadata(emission)
                ),
                None,
            )
            if previous is None:
                merged.append(emission)
            else:
                for name in emission.submission.payloads:
                    left, structure = _payload_leaves(previous.arguments[name])
                    right, other_structure = _payload_leaves(emission.arguments[name])
                    if structure != other_structure:
                        raise ValueError("Access energy contributions must share their payload structure")
                    summed_nodes = [graph.call_function(operator.add, pair) for pair in zip(left, right, strict=True)]
                    previous.arguments[name] = pytree.tree_unflatten(summed_nodes, structure)
        return merged
    if scope.kind != "chunk" or scope.emissions:
        raise ValueError("A chunk scope accepts records only from its positional groups")
    if not scope.groups:
        return []
    count = len(scope.groups[0])
    if any(len(group) != count for group in scope.groups):
        raise ValueError("All chunk groups must emit the same record schema")
    result: list[_Emission] = []
    for index in range(count):
        peers = [group[index] for group in scope.groups]
        first = peers[0]
        if any(peer.submission != first.submission or _metadata(peer) != _metadata(first) for peer in peers[1:]):
            raise ValueError("Chunk groups disagree on record identity or metadata")
        arguments = dict(first.arguments)
        for name in first.submission.payloads:
            leaves = [_payload_leaves(peer.arguments[name]) for peer in peers]
            if any(structure != leaves[0][1] for _, structure in leaves[1:]):
                raise ValueError("Chunk groups disagree on record payload structure")
            joined = []
            for values in zip(*(item[0] for item in leaves), strict=True):
                value = values[0] if len(values) == 1 else graph.call_function(torch.cat, (list(values),), {"dim": 0})
                joined.append(graph.call_function(_restore_positions, (value, scope.shape)))
            arguments[name] = pytree.tree_unflatten(joined, leaves[0][1])
        result.append(_Emission(first.submission, arguments))
    return result


def _payload_leaves(value: Argument) -> tuple[list[Argument], pytree.TreeSpec]:
    return pytree.tree_flatten(value)


def recording_backend(
    module: GraphModule, example_inputs: list[Tensor], *, backend: str | None = None
) -> Callable[..., Sequence[Argument]]:
    """Lower only NeuroX record scopes before invoking a PyTorch backend.

    Submissions become immutable hidden outputs. The returned execution wrapper
    submits them after successful computation and returns only the original
    graph outputs. Raw PyTorch loops and dynamic control flow cannot transport
    hidden records. Register this backend on an enclosing `torch.compile` when
    that enclosing region contains recorded NeuroX loops.
    """
    emissions = _transform(module, loop_body=False)
    graph = module.graph
    output = next(node for node in graph.nodes if node.op == "output")
    original = tuple(output.args[0])
    with graph.inserting_before(output):
        for emission in emissions:
            rank_name = emission.submission.leading_rank
            if rank_name is not None and emission.arguments[rank_name] is not None:
                rank = emission.arguments[rank_name]
                if not isinstance(rank, int):
                    raise ValueError("Observation rank must be fixed while compiling")

                def normalize(value: Node, rank: int = rank) -> Node:
                    return graph.call_function(_reduce_observations, (value, rank))

                normalized = _map_payload(emission, normalize)
                emission.arguments = normalized.arguments
                emission.arguments[rank_name] = None
    extra: list[Node] = []
    positions: dict[Node, int] = {}

    def retain(value: Node) -> Node:
        if value not in positions:
            positions[value] = len(original) + len(extra)
            extra.append(value)
        return value

    for emission in emissions:
        map_arg(emission.arguments, retain)
    output.args = (original + tuple(extra),)
    graph.lint()
    module.recompile()
    core: Callable[..., Sequence[Argument]]
    if backend == "eager":
        core = module.forward
    else:
        from torch._dynamo.backends.registry import lookup_backend

        core = lookup_backend("inductor" if backend is None else backend)(module, example_inputs)
    if not emissions:
        return core

    def execute(*args: Tensor) -> Sequence[Argument]:
        result = core(*args)
        for emission in emissions:
            arguments = map_arg(emission.arguments, lambda value: result[positions[value]])
            emission.submission.deliver(**arguments)
        return result[: len(original)]

    return execute


class _CompileDecorator(Protocol):
    def __call__[**P, R](self, function: Callable[P, R]) -> Callable[P, R]: ...


@overload
def compile_recording[**P, R](
    function: Callable[P, R],
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | None = None,
) -> Callable[P, R]: ...


@overload
def compile_recording(
    function: None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | None = None,
) -> _CompileDecorator: ...


def compile_recording(
    function: Callable[..., Any] | None = None,
    *,
    fullgraph: bool = False,
    dynamic: bool | None = None,
    backend: str | None = None,
) -> Callable[..., Any]:
    """Compile a callable with NeuroX's scoped side-channel collection.

    Numerical results and callable signatures are preserved. Recorder contexts
    belong outside compiled execution; their snapshots become visible after
    each successful compiled call. Profiling enablement and record schemas
    specialize the graph, while tensor values and recorder identities remain
    runtime inputs. The supported collection path is inference/no-grad.
    """

    def compiler(module: GraphModule, example_inputs: list[Tensor]) -> Callable[..., Sequence[Argument]]:
        return recording_backend(module, example_inputs, backend=backend)

    compiled: Callable[..., Any] = torch.compile(function, backend=compiler, fullgraph=fullgraph, dynamic=dynamic)
    return compiled
