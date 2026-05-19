"""PT2E QAT for LeNet-5 on MNIST (NeuroX-grid ranges, modern API).

Replaces the deprecated ``torch.ao.quantization.prepare_qat`` /
``convert`` eager-mode flow with the current PT2E flow from
``torchao.quantization.pt2e``.  The model is exported with
``torch.export.export_for_training`` to a graph module, a custom
``Quantizer`` annotates each ``conv2d`` / ``linear`` node with the
NeuroX-matching spec, ``prepare_qat_pt2e`` inserts fake-quant +
observers, fine-tuning runs, and ``convert_pt2e`` lowers to a
quantized graph.

Default quantization grid (matches the NeuroX physical crossbar):

    activation: per-tensor affine, ``[--x-qmin, --x-qmax]``
                default ``[0, 15]`` -> 16 levels (4-bit unsigned grid)
    weight:     per-channel symmetric, ``[-w_qmax, w_qmax]``
                default ``[-63, 63]`` -> 127 levels per output channel
                (fits in ``int8``; widen via ``--w-qmax`` if desired)

The QAT-prepared model holds float weights plus the per-channel weight
fake-quant scale and per-tensor activation scale/zero-point learned
during fine-tuning.  The script saves the converted graph state_dict
together with the qat config metadata for downstream NeuroX usage.
"""

# ruff: noqa: T201

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchao.quantization.pt2e import (
    allow_exported_model_train_eval,
)
from torchao.quantization.pt2e.fake_quantize import FusedMovingAvgObsFakeQuantize
from torchao.quantization.pt2e.observer import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)

from example.common.pt2e import pt2e_to_neurox_state
from example.lenet.data import create_mnist_dataloader
from example.lenet.model import LeNet5

# ---------------------------------------------------------------------------
# Custom NeuroX-grid quantizer
# ---------------------------------------------------------------------------


class NeuroXQuantizer(Quantizer):
    """Annotate ``conv2d`` / ``linear`` nodes with NeuroX-grid specs.

    Activation: per-tensor affine, ``[x_qmin, x_qmax]``, ``uint8`` storage.
    Weight:     per-channel symmetric, ``[-w_qmax, w_qmax]``, ``int8`` storage
                (so ``w_qmax`` must satisfy ``w_qmax <= 127``).
    """

    _CONV_TARGETS = (
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv3d.default,
    )
    _LINEAR_TARGETS = (torch.ops.aten.linear.default,)

    def __init__(self, x_qmin: int, x_qmax: int, w_qmax: int) -> None:
        super().__init__()
        if not (0 <= x_qmin < x_qmax <= 255):
            raise ValueError(f"activation range [{x_qmin}, {x_qmax}] must satisfy 0 <= qmin < qmax <= 255")
        if not (0 < w_qmax <= 127):
            raise ValueError(f"weight bound w_qmax={w_qmax} must satisfy 0 < w_qmax <= 127 to fit in int8")

        self.act_spec = QuantizationSpec(
            dtype=torch.uint8,
            quant_min=x_qmin,
            quant_max=x_qmax,
            qscheme=torch.per_tensor_affine,
            is_dynamic=False,
            observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=x_qmin,
                quant_max=x_qmax,
                dtype=torch.uint8,
                qscheme=torch.per_tensor_affine,
            ),
        )
        self.weight_spec = QuantizationSpec(
            dtype=torch.int8,
            quant_min=-w_qmax,
            quant_max=w_qmax,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=-w_qmax,
                quant_max=w_qmax,
                dtype=torch.int8,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            ),
        )

    def annotate(self, model: GraphModule) -> GraphModule:
        """Tag every conv / linear node with input, weight, and output specs."""
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in self._CONV_TARGETS + self._LINEAR_TARGETS:
                continue
            if node.meta.get("quantization_annotation", None) is not None:
                continue
            input_qspec_map: dict = {}
            input_act = node.args[0]
            if isinstance(input_act, torch.fx.Node):
                input_qspec_map[input_act] = self.act_spec
            weight = node.args[1]
            if isinstance(weight, torch.fx.Node):
                input_qspec_map[weight] = self.weight_spec
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=self.act_spec,
                _annotated=True,
            )
        return model

    def validate(self, model: GraphModule) -> None:
        """No structural validation needed for this minimal quantizer."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return top-1 accuracy of ``model`` on ``loader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            correct += model(images).argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PT2E QAT for LeNet-5 on MNIST (NeuroX-grid ranges)")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="MNIST root directory")
    parser.add_argument(
        "--float-checkpoint",
        type=Path,
        required=True,
        help="Pretrained float state_dict produced by example.lenet.train",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Output QAT graph state_dict path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for QAT fine-tuning")
    parser.add_argument("--x-qmin", type=int, default=0, help="Activation integer minimum")
    parser.add_argument("--x-qmax", type=int, default=15, help="Activation integer maximum")
    parser.add_argument(
        "--w-qmax",
        type=int,
        default=63,
        help="Symmetric weight integer bound; range is [-w_qmax, w_qmax]; must be <= 127",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- 1. Load pretrained float weights ---
    # The float reference model is kept around after export; the extractor
    # needs access to the original ``nn.Linear`` / ``nn.Conv2d`` modules by
    # name to fold bias and re-quantize weights against the learned scales.
    float_state = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)
    float_reference = LeNet5()
    float_reference.load_state_dict(float_state)

    model = LeNet5()
    model.load_state_dict(float_state)
    model = model.to(device).eval()
    print(f"Loaded float checkpoint: {args.float_checkpoint}")

    # --- 2. Export to a graph module suitable for QAT prepare ---
    # ``torch.export.export`` in torch 2.11 unified the training and
    # inference export paths; the resulting graph module can be put in
    # train mode after ``prepare_qat_pt2e`` inserts the fake-quant ops.
    # Use ``Dim.AUTO`` on the batch axis (and a representative batch>1
    # in the example input) so the graph stays polymorphic across the
    # train/val batch sizes used by the loaders.
    example_inputs = (torch.randn(2, 1, 28, 28, device=device),)
    exported = torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=({0: torch.export.Dim.AUTO},),
    ).module()

    # --- 3. Build the NeuroX-grid quantizer and prepare for QAT ---
    quantizer = NeuroXQuantizer(args.x_qmin, args.x_qmax, args.w_qmax)
    prepared = prepare_qat_pt2e(exported, quantizer)
    # Patch train()/eval() onto the exported graph so we can drive the
    # standard nn.Module training loop without special-casing.
    allow_exported_model_train_eval(prepared)
    prepared.train()
    print(
        f"QAT grid: activation [{args.x_qmin}, {args.x_qmax}] "
        f"({args.x_qmax - args.x_qmin + 1} levels, per-tensor affine, uint8), "
        f"weight [{-args.w_qmax}, {args.w_qmax}] "
        f"({2 * args.w_qmax + 1} levels, per-channel symmetric, int8)"
    )

    # --- 4. Fine-tune with fake quantization ---
    train_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="train", shuffle=True)
    val_loader = create_mnist_dataloader(args.dataset_dir, args.batch_size, device, split="val")

    optimizer = torch.optim.SGD(
        prepared.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        prepared.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            output = prepared(images)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += output.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total
        val_acc = _validate(prepared, val_loader, device)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(prepared.state_dict())
            marker = " *best*"

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={total_loss / len(train_loader):.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}{marker}"
        )

    if best_state is not None:
        prepared.load_state_dict(best_state)
    print(f"Best fake-quant val_acc: {best_acc:.4f}")

    # --- 5. Extract BEFORE convert (convert mutates the prepared graph) ---
    # ``convert_pt2e`` replaces the FakeQuantize call_module nodes with
    # ``quantize_per_tensor`` / ``dequantize_per_tensor`` call_function
    # ops in place, which drops the scale/zero_point buffers we need for
    # extraction.  Pull the NeuroX-flat state_dict out of ``prepared``
    # first, then run ``convert_pt2e`` purely for the final reporting.
    prepared.eval()
    flat_state = pt2e_to_neurox_state(prepared, float_reference)

    converted = convert_pt2e(prepared)
    allow_exported_model_train_eval(converted)
    final_acc = _validate(converted, val_loader, device)
    print(f"Converted graph val_acc: {final_acc:.4f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "neurox_flat",
            "state_dict": flat_state,
            "qat_config": {
                "x_qmin": args.x_qmin,
                "x_qmax": args.x_qmax,
                "w_qmax": args.w_qmax,
            },
        },
        args.checkpoint,
    )
    print(f"Saved NeuroX-flat QAT checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
