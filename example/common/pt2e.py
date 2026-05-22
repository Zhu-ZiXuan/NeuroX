"""PT2E → NeuroX flat state_dict extractor (example-local helper).

Consumes a ``torchao.quantization.pt2e`` QAT-prepared graph module and
produces a flat ``state_dict`` that ``neurox.api.build_evaluator`` can
load directly.  This lives under ``example/common/`` because pt2e is
one specific QAT front-end; NeuroX core is format-agnostic and only
sees the ``neurox_flat`` state_dict that this helper emits.

Required inputs:

- ``prepared``: the ``GraphModule`` returned by ``prepare_qat_pt2e``.
  It carries FakeQuantize submodules whose ``scale`` / ``zero_point``
  buffers hold the learned activation and weight statistics.  Using
  the prepared graph (rather than the post-``convert_pt2e`` graph)
  keeps those numbers reachable as regular tensors instead of Python
  constants baked into graph args.
- ``float_model``: the original ``nn.Module`` the graph was exported
  from.  Supplies the float weights and biases (by named-module
  lookup) and the qualified layer names used in the output keys.

Output schema ("neurox_flat"): for every quantized layer the dict
contains twelve entries keyed ``f"{layer_name}.{field}"`` where the
fields match the buffers declared on ``QuantLinear`` /
``QuantConv2d``::

    weight_int           int8    [C_out, ...]
    bias_int             int32   [C_out]   (pre-folded with the
                                            input zero-point cross-term)
    rescale_multiplier   int32   [C_out]
    rescale_rshift       int32   [C_out]
    output_zero_point    int32   [1]
    output_qmin          int32   scalar
    output_qmax          int32   scalar
    input_scale          float32 [1]
    input_zero_point     int32   [1]
    input_qmin           int32   scalar
    input_qmax           int32   scalar
    output_scale         float32 [1]

Non-quantized state from ``float_model`` (e.g. ``BatchNorm`` running
stats) is copied through unchanged, so the returned dict is a complete
state_dict for the replaced model.
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx import GraphModule, Node

from neurox.operator.qat_util import derive_multiplier_and_shift_tensor

_LINEAR_TARGETS = (torch.ops.aten.linear.default,)
_CONV_TARGETS = (
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.conv3d.default,
)


def _layer_name_from_node(node: Node) -> str:
    """Pluck the last ``nn_module_stack`` entry's qualified name.

    pt2e preserves ``node.meta["nn_module_stack"]`` as a dict mapping
    an export-level stack id to ``(qualified_name, type_str)`` tuples.
    The last entry names the innermost module the op came from, which
    for a simple model like LeNet is directly the ``nn.Linear`` /
    ``nn.Conv2d`` attribute name (``"conv1"``, ``"fc1"``, ...).
    """
    stack = node.meta.get("nn_module_stack")
    if not stack:
        raise RuntimeError(f"node {node.name!r} has no nn_module_stack metadata; cannot map it back to the float model")
    qualified_name, _type_str = next(reversed(stack.values()))
    return qualified_name


def _read_scale_zp(fq: nn.Module) -> tuple[Tensor, Tensor]:
    """Read ``(scale, zero_point)`` from a FakeQuantize submodule.

    Both attributes are present on ``FakeQuantize`` /
    ``FusedMovingAvgObsFakeQuantize`` and on the eager-mode observers
    from torchao; the fused module also exposes ``.activation_post_process``
    but its top-level ``scale`` / ``zero_point`` track the latest observer
    output so we can read them directly.
    """
    scale = fq.scale.detach().clone().to(torch.float32).reshape(-1)
    zero_point = fq.zero_point.detach().clone().to(torch.int32).reshape(-1)
    return scale, zero_point


def _resolve_fq(graph: GraphModule, arg: Any) -> nn.Module:  # noqa: ANN401
    """Follow a ``get_attr`` / call_module node to the attached fake-quant.

    The pt2e prepared graph routes each conv/linear through a
    ``call_module`` node that invokes a FakeQuantize submodule attached
    to the root graph module.  Given one of the node's args, return the
    submodule it targets.
    """
    if not isinstance(arg, Node):
        raise TypeError(f"expected fx.Node for FakeQuantize arg, got {type(arg).__name__}")
    target = arg.target
    if not isinstance(target, str):
        raise TypeError(f"expected string target, got {target!r}")
    return graph.get_submodule(target)


def _find_output_fq(graph: GraphModule, compute_node: Node) -> nn.Module | None:
    """Find the FakeQuantize that observes ``compute_node``'s output."""
    for user in compute_node.users:
        if user.op != "call_module":
            continue
        sub = graph.get_submodule(user.target) if isinstance(user.target, str) else None
        if sub is None:
            continue
        # FakeQuantize submodules expose ``.scale`` and ``.zero_point``.
        if hasattr(sub, "scale") and hasattr(sub, "zero_point"):
            return sub
    return None


def _quantize_weight(weight_float: Tensor, weight_scale: Tensor, w_qmin: int, w_qmax: int) -> Tensor:
    """Per-channel symmetric quantize a weight tensor to int8."""
    # Broadcast scale from [C_out] to [C_out, 1, 1, ...] matching weight rank.
    shape = [weight_scale.shape[0]] + [1] * (weight_float.ndim - 1)
    sw = weight_scale.view(shape).to(weight_float.device)
    q = torch.round(weight_float / sw).clamp(w_qmin, w_qmax)
    return q.to(torch.int8)


def _fold_bias(
    bias_float: Tensor | None,
    w_int: Tensor,
    input_scale: Tensor,
    input_zero_point: Tensor,
    weight_scale: Tensor,
    *,
    out_features: int,
    rescale_factor: float,
) -> Tensor:
    """Fold ``(bias / (sx * sw) - zp_x * sum_K(w_int)) / rescale_factor`` into int32.

    The folded bias is added into the int MAC accumulator before the
    operator's inline multiply-shift requantize.  The cross-term absorbs
    the activation zero-point shift; the division by ``rescale_factor``
    compensates for the fact that ``y_agg`` emitted by the crossbar is
    already in ADC-code scale (``rf`` codes ≈ 1 ideal-integer state),
    so adding an ideal-scale bias before the requantize would over-weight
    it by ``rf``.  See ``neurox.operator.linear.derive_layer_int_params``
    for the algebraic derivation.
    """
    device = w_int.device
    # Sum integer weights across all non-out-channel dims.
    k_axes = tuple(range(1, w_int.ndim))
    w_sum = w_int.to(torch.int64).sum(dim=k_axes) if k_axes else w_int.to(torch.int64)

    sx = input_scale.to(torch.float64).to(device)
    zp_x = input_zero_point.to(torch.float64).to(device)
    sw = weight_scale.to(torch.float64).to(device)
    rf = float(rescale_factor) if rescale_factor != 0.0 else 1.0

    if bias_float is not None:
        scale = (sx * sw).clamp(min=1e-30)
        bias_ideal = bias_float.to(torch.float64).to(device) / scale
    else:
        bias_ideal = torch.zeros(out_features, dtype=torch.float64, device=device)

    cross = zp_x * w_sum.to(torch.float64)
    folded = torch.round((bias_ideal - cross) / rf)
    int32_info = torch.iinfo(torch.int32)
    return folded.clamp(min=int32_info.min, max=int32_info.max).to(torch.int32)


def _trace_weight_source(weight_fq_node: Node) -> Node:
    """Walk the weight FakeQuantize's sole tensor input back to its get_attr."""
    input_nodes = list(weight_fq_node.all_input_nodes)
    if len(input_nodes) != 1:
        raise RuntimeError(
            f"expected single tensor input to weight FakeQuantize {weight_fq_node.name!r}, got {len(input_nodes)}"
        )
    src = input_nodes[0]
    if src.op != "get_attr":
        raise RuntimeError(f"weight FakeQuantize {weight_fq_node.name!r} input must be get_attr, got op={src.op!r}")
    return src


def _fetch_trained_tensor(graph: GraphModule, target: str) -> Tensor:
    """Fetch a named parameter/buffer from ``graph`` (supports dotted names)."""
    params = dict(graph.named_parameters())
    if target in params:
        return params[target].detach().cpu()
    buffers = dict(graph.named_buffers())
    if target in buffers:
        return buffers[target].detach().cpu()
    raise KeyError(f"no parameter or buffer named {target!r} in the prepared graph")


def _extract_layer(
    graph: GraphModule,
    node: Node,
    layer_name: str,
    rescale_factor: float,
) -> dict[str, Tensor]:
    """Build the 10-buffer NeuroX-flat entries for one conv/linear layer.

    Pulls the trained float weight/bias directly from ``graph`` (the
    pt2e prepared graph whose parameters were updated by QAT fine-tuning)
    rather than from any pre-QAT float reference.  ``rescale_factor``
    is folded into both the bias and the ``(mult, rshift)`` pair so the
    operator's inline requantize covers the full ADC-correction +
    output-rescale math in one shot (see
    ``derive_layer_int_params`` docstring for the derivation).
    """
    # --- read fake-quant stats from the prepared graph ---
    input_fq = _resolve_fq(graph, node.args[0])
    weight_fq_node = node.args[1]
    weight_fq = _resolve_fq(graph, weight_fq_node)
    output_fq = _find_output_fq(graph, node)
    if output_fq is None:
        raise RuntimeError(f"layer {layer_name!r}: no output FakeQuantize found among graph users")

    input_scale, input_zp = _read_scale_zp(input_fq)
    weight_scale, _ = _read_scale_zp(weight_fq)  # symmetric: zp = 0
    output_scale, output_zp = _read_scale_zp(output_fq)

    # --- pull quant bounds from the fake-quant modules ---
    x_qmin = int(input_fq.quant_min)
    x_qmax = int(input_fq.quant_max)
    y_qmin = int(output_fq.quant_min)
    y_qmax = int(output_fq.quant_max)
    w_qmin = int(weight_fq.quant_min)
    w_qmax = int(weight_fq.quant_max)
    assert w_qmin == -w_qmax, f"weight range must be symmetric, got [{w_qmin}, {w_qmax}]"

    # --- quantize QAT-trained weight + fold QAT-trained bias ---
    # Pull the trained float weight from the prepared graph, not from the
    # pre-QAT float reference: pt2e updates the underlying float parameters
    # during fine-tuning, so the reference model would produce mis-quantized
    # weights that disagree with the scales recorded by the weight FQ.
    weight_node = _trace_weight_source(weight_fq_node)
    weight_float = _fetch_trained_tensor(graph, weight_node.target)

    weight_scale_cpu = weight_scale.cpu()
    w_int = _quantize_weight(weight_float, weight_scale_cpu, w_qmin, w_qmax)

    # Bias (optional): ``aten.linear.default`` takes (x, w, bias),
    # ``aten.conv2d.default`` takes (x, w, bias, stride, padding, ...).
    # In both cases args[2] is the bias node (or None / missing).
    bias_float: Tensor | None = None
    if len(node.args) > 2 and node.args[2] is not None:
        bias_node = node.args[2]
        if isinstance(bias_node, Node) and bias_node.op == "get_attr":
            bias_float = _fetch_trained_tensor(graph, bias_node.target)

    out_features = w_int.shape[0]
    bias_int = _fold_bias(
        bias_float,
        w_int,
        input_scale.cpu(),
        input_zp.cpu(),
        weight_scale_cpu,
        out_features=out_features,
        rescale_factor=rescale_factor,
    )

    # --- derive the fixed-point requantizer pair ---
    # Multiply ``(s_x * s_w / s_y)`` by ``rescale_factor`` so the macro
    # can fuse ADC-scale → ideal-scale conversion into its single
    # requantize step.
    rf = float(rescale_factor) if rescale_factor != 0.0 else 1.0
    combined_scale = (
        input_scale.cpu().to(torch.float64) * weight_scale_cpu.to(torch.float64) * rf
    ) / output_scale.cpu().to(torch.float64).clamp(min=1e-30)
    rescale_multiplier, rescale_rshift = derive_multiplier_and_shift_tensor(combined_scale.to(torch.float32))

    return {
        f"{layer_name}.weight_int": w_int,
        f"{layer_name}.bias_int": bias_int,
        f"{layer_name}.rescale_multiplier": rescale_multiplier,
        f"{layer_name}.rescale_rshift": rescale_rshift,
        f"{layer_name}.output_zero_point": output_zp.reshape(1).to(torch.int32).cpu(),
        f"{layer_name}.output_qmin": torch.tensor(y_qmin, dtype=torch.int32),
        f"{layer_name}.output_qmax": torch.tensor(y_qmax, dtype=torch.int32),
        f"{layer_name}.input_scale": input_scale.reshape(1).to(torch.float32).cpu(),
        f"{layer_name}.input_zero_point": input_zp.reshape(1).to(torch.int32).cpu(),
        f"{layer_name}.input_qmin": torch.tensor(x_qmin, dtype=torch.int32),
        f"{layer_name}.input_qmax": torch.tensor(x_qmax, dtype=torch.int32),
        f"{layer_name}.output_scale": output_scale.reshape(1).to(torch.float32).cpu(),
    }


def pt2e_to_neurox_state(
    prepared: GraphModule,
    float_model: nn.Module | None = None,
    rescale_factor: float = 1.0,
) -> dict[str, Tensor]:
    """Extract a NeuroX-flat state_dict from a pt2e QAT-prepared graph.

    Walks the prepared graph's FX nodes in source order, extracts per-layer
    activation / weight fake-quant stats, quantizes the *QAT-trained* float
    weights (pulled from ``prepared`` itself — not from any pre-QAT
    reference), folds bias + zero-point, and derives ``(multiplier, rshift)``.

    If ``float_model`` is provided, any non-quantized state it carries
    (``BatchNorm`` running buffers, unquantized parameters, etc.) is copied
    through unchanged so the returned dict can be loaded directly into a
    replaced model via ``load_state_dict(..., strict=False)``.

    Args:
        prepared: GraphModule returned by ``prepare_qat_pt2e``; must still
            carry its FakeQuantize submodules (do NOT pass the post-convert
            graph — activation scales there are baked as graph-arg constants).
        float_model: Optional reference to the original ``nn.Module`` that
            was exported to produce ``prepared``.  Used only to forward
            non-quantized state (e.g. BN running stats).  Weights and biases
            for quantized layers are always read from ``prepared``.
        rescale_factor: Target macro's ``output_rescale_factor``
            (``N_states / N_codes``).  Multiplied into ``(mult, rshift)``
            and divided into the folded bias so the operator's inline
            requantize produces correctly-scaled output.  Pass
            ``1.0`` (default) for macros whose ADC already covers the
            full ideal state range (e.g. ``IdealXbarMacro``).

    Returns:
        Flat state_dict ready for ``model.load_state_dict`` on a model whose
        ``nn.Linear`` / ``nn.Conv2d`` modules have already been replaced with
        ``QuantLinear`` / ``QuantConv2d``.
    """
    state: dict[str, Tensor] = {}
    quantized_prefixes: set[str] = set()

    for node in prepared.graph.nodes:
        if node.op != "call_function" or node.target not in (*_LINEAR_TARGETS, *_CONV_TARGETS):
            continue

        layer_name = _layer_name_from_node(node)
        state.update(_extract_layer(prepared, node, layer_name, rescale_factor))
        quantized_prefixes.add(f"{layer_name}.")

    # Copy through non-quantized state (BN buffers, unquantized params, etc.)
    # so the returned dict is a complete state_dict for the replaced model.
    if float_model is not None:
        for key, value in float_model.state_dict().items():
            if any(key.startswith(p) for p in quantized_prefixes):
                continue
            state[key] = value.detach().cpu().clone()

    return state
