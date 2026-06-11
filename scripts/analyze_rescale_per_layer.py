"""Per-layer rescale_factor diagnostic for LeNet.

For each Conv2d / Linear layer in the trained QuantLeNet5:
  1. Compute the effective row count (in_features for Linear; kH·kW·in_C for Conv2d).
  2. Sample a real MNIST batch and capture the per-cycle integer dot products
     produced inside the macro pipeline (pre-ADC, pre-rescale).
  3. Report typical / max |dot| and compare to the macro-applied rescale_factor.
  4. Estimate the "optimal" rescale that would saturate ADC at ~max |dot|.

Identifies which layers are underfilled and quantifies the rescale mismatch.

Output to log/lenet_rescale_audit.log.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import neurox  # noqa: E402
from example.common import build_macro_factory  # noqa: E402
from example.lenet.data import create_mnist_dataloader  # noqa: E402
from example.lenet.model_quant import QuantLeNet5  # noqa: E402


def main() -> None:
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    factory = build_macro_factory(Path("example/lenet/macro_with_ideal_xbar.toml"), ideal_xbar=False)
    model = QuantLeNet5(factory)
    ckpt = torch.load("weight/lenet_hat_ternary.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    neurox.fabricate_model(model)
    neurox.program_model(model)
    model = model.to(device).eval()

    loader = create_mnist_dataloader("dataset/mnist", 128, device, split="val")
    images, _ = next(iter(loader))
    images = images.to(device)

    print("=" * 80)
    print("LeNet per-layer rescale audit")
    print("=" * 80)

    layers = [("conv1", model.conv1), ("conv2", model.conv2), ("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]

    # Capture pre-rescale per-cycle dot products by patching xbar.vec_mat_mul
    # to record its IDEAL integer matmul output before ADC quantization.
    captured: dict[str, list[torch.Tensor]] = {n: [] for n, _ in layers}

    def make_probe(name: str, original_fn):  # noqa: ANN001
        def probed(x, *, adc_operation_point):  # noqa: ANN001
            # Replicate IdealXbar.vec_mat_mul's pre-ADC dot product so we
            # can inspect the raw range without quantization-induced clipping.
            xbar = original_fn.__self__
            digits = xbar.digits.to(torch.int64)
            digit_weights = xbar.digit_weights.to(torch.int64).view(*([1] * (digits.ndim - 2)), -1, 1)
            w = (digits * digit_weights).sum(dim=-2)
            xx = x.to(torch.int64).unsqueeze(-2)
            full = torch.broadcast_shapes(w.shape, xx.shape)
            dot = (xx.expand(full) * w.expand(full)).sum(dim=-1)
            captured[name].append(dot.flatten().cpu())
            return original_fn(x, adc_operation_point=adc_operation_point)

        return probed

    for name, layer in layers:
        layer.macro.xbar.vec_mat_mul = make_probe(name, layer.macro.xbar.vec_mat_mul)

    with torch.no_grad():
        _ = model(images)

    for name, layer in layers:
        macro = layer.macro
        xbar = macro.xbar
        weight_int = layer.weight_int
        rescale = macro.adc_rescale_factor(layer.adc_operation_point)

        if hasattr(layer, "in_features"):
            eff_rows = layer.in_features
        else:  # conv2d
            eff_rows = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        col_per_tile = xbar.col_num
        row_per_tile = xbar.row_num
        n_row_tiles = (eff_rows + row_per_tile - 1) // row_per_tile
        last_tile_fill = eff_rows - (n_row_tiles - 1) * row_per_tile

        w_sparsity = (weight_int == 0).float().mean().item()
        w_nonzero_per_out = (weight_int != 0).sum(dim=tuple(range(1, weight_int.ndim))).float().mean().item()

        dots = torch.cat(captured[name]) if captured[name] else torch.zeros(0)
        if dots.numel() > 0:
            d = dots.abs().float()
            if d.numel() > 16_000_000:
                d = d[torch.randperm(d.numel())[:16_000_000]]
            max_dot = d.max().item()
            p99_dot = torch.quantile(d, 0.99).item()
            std_dot = dots.float().std().item()
            n_zero_code = (dots.abs() < rescale * 0.5).float().mean().item() * 100
            n_satur = (dots.abs() > rescale * 7.5).float().mean().item() * 100
        else:
            max_dot = p99_dot = std_dot = n_zero_code = n_satur = float("nan")

        opt_rescale = max_dot / 7.0 if max_dot > 0 else float("nan")
        eff_bits = math.log2(max_dot / rescale + 1) if max_dot / rescale > 0 else float("nan")

        print(f"\n--- {name} ---")
        print(f"  effective_rows = {eff_rows} (per-tile fill: {n_row_tiles - 1} × full + 1 × {last_tile_fill}/{row_per_tile})")
        print(f"  weight sparsity (zero ratio) = {w_sparsity:.3f}, avg non-zero per output = {w_nonzero_per_out:.1f}")
        print(f"  per-cycle |dot| stats: max={max_dot:.1f}, p99={p99_dot:.1f}, std={std_dot:.2f}")
        print(f"  current rescale_factor = {rescale:.3f}")
        print(f"  → code = floor(dot/rescale); typical max|code| ≈ {max_dot / rescale:.2f} (4-bit signed range [-8, 7])")
        print(f"  → effective ADC bits used ≈ {eff_bits:.2f} of 4 available")
        print(f"  → % cycles producing code 0 (|dot| < rescale/2): {n_zero_code:.1f}%")
        print(f"  → % cycles producing saturated code (|dot| > rescale*7.5): {n_satur:.1f}%")
        print(f"  → suggested rescale to use full 4-bit range: {opt_rescale:.3f}  (over-rescale factor: {rescale / opt_rescale:.2f}x)")


if __name__ == "__main__":
    main()
