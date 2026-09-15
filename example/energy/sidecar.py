"""Evaluate linear and convolution energy without replacing numerical outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

from neurox import Profiler, fabricate, stamp_names

from .factory import UnitFactory


def integer_codes(value: Tensor, bits: int) -> Tensor:
    """Preserve bounded integer inputs; otherwise apply symmetric tensor quantization.

    Floating operands use a tensor-wide absolute-maximum scale. This affects
    only the measured hardware workload; the original operator keeps its input.
    """
    value = value.detach()
    limit = (1 << bits) - 1
    rounded = value.round()
    if bool(((value == rounded) & (value.abs() <= limit)).all()):
        return rounded.to(torch.int32)
    scale = value.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / limit
    return (value / scale).round().clamp(-limit, limit).to(torch.int32)


def _pair(value: int | Sequence[int]) -> tuple[int, ...]:
    return (value, value) if isinstance(value, int) else tuple(value)


def convolution_chunks(
    input: Tensor,
    *,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    max_positions: int,
) -> Iterator[Tensor]:
    """Yield input windows whose disjoint output rectangles cover one convolution.

    Padding is applied once; each crop retains the complete strided, dilated
    kernel halo and is evaluated by a zero-padding convolution unit.
    """
    ph, pw = padding
    input = F.pad(input, (pw, pw, ph, ph)) if ph or pw else input
    kh, kw = ((k - 1) * d + 1 for k, d in zip(kernel, dilation, strict=True))
    sh, sw = stride
    oh = (input.shape[-2] - kh) // sh + 1
    ow = (input.shape[-1] - kw) // sw + 1
    columns = min(ow, max_positions)
    rows = max(1, max_positions // columns)
    for h in range(0, oh, rows):
        height = min(rows, oh - h)
        for w in range(0, ow, columns):
            width = min(columns, ow - w)
            yield input[..., h * sh : (h + height - 1) * sh + kh, w * sw : (w + width - 1) * sw + kw]


@dataclass
class Measurement:
    calls: int = 0
    dynamic_energy__fJ: float = 0.0


class EnergySidecar(TorchFunctionMode):
    """Observe actual functional operators and return their original results.

    Enter after loading weights and moving the model. Grouped convolutions
    use separate units per group; Conv1d is represented as a height-one Conv2d.
    Units and records live outside the numerical model. Each hardware call has
    its own profiler context so compiled submissions start with an empty ledger.
    """

    def __init__(
        self, model: nn.Module, factory: UnitFactory, *, batch_chunk: int = 1, spatial_chunk: int = 32
    ) -> None:
        super().__init__()
        self.model = model
        self.factory = factory
        self._modules = dict(model.named_modules())
        self.batch_chunk = batch_chunk
        self.spatial_chunk = spatial_chunk
        self.units = nn.ModuleDict()
        self.weights: dict[str, Tensor] = {}
        self.measurements: dict[str, Measurement] = defaultdict(Measurement)
        self._stack: list[tuple[str, int]] = []
        self._hooks = []
        self._busy = False
        self._handled_depth = 0

    def __enter__(self) -> Self:
        for name, module in self.model.named_modules():
            self._hooks.append(module.register_forward_pre_hook(self._before(name)))
            self._hooks.append(module.register_forward_hook(self._after, always_call=True))
        return super().__enter__()

    def __exit__(self, *args: object) -> None:
        try:
            return super().__exit__(*args)
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._stack.clear()

    def _before(self, name: str) -> Callable:
        def hook(module: nn.Module, args: tuple) -> None:
            self._stack.append((name, 0))
            # SOUL quantized wrappers may implement their operator through tile
            # matmuls rather than F.linear/F.conv2d; measure their logical seam.
            if module.__class__.__name__ in ("QLinear", "QConv") and hasattr(module, "is_spike"):
                self._handled_depth += 1
                with self._measurement():
                    x = args[0]
                    spike = module.mode == "adc" and module.is_spike and module.sparse
                    input_bits = 1 if spike else self.factory.input_bits
                    weight_bits = self.factory.weight_bits if module.mode == "fp" else 1
                    if spike:
                        x = (x / module.act_alpha.clamp_min(1e-8)).round().clamp(0, 1)
                    if module.__class__.__name__ == "QLinear":
                        w = module.weight if module.mode == "fp" else module.weight.sign()
                        self.observe(name, x, w, input_bits=input_bits, weight_bits=weight_bits)
                    else:
                        c = module.conv
                        w = c.weight if module.mode == "fp" else c.weight.sign()
                        self.observe(
                            name,
                            x,
                            w,
                            stride=c.stride,
                            padding=c.padding,
                            dilation=c.dilation,
                            groups=c.groups,
                            input_bits=input_bits,
                            weight_bits=weight_bits,
                        )

        return hook

    def _after(self, module: nn.Module, args: tuple, output: object) -> None:
        if module.__class__.__name__ in ("QLinear", "QConv") and hasattr(module, "is_spike"):
            self._handled_depth -= 1
        self._stack.pop()

    @contextmanager
    def _measurement(self) -> Iterator[None]:
        previous = self._busy
        self._busy = True
        try:
            stance = torch.compiler.set_stance("force_eager")
            with torch.no_grad(), stance:
                yield
        finally:
            self._busy = previous

    def __torch_function__(self, func: Callable, types: tuple, args: tuple = (), kwargs: dict | None = None) -> object:
        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        if not self._busy and not self._handled_depth and func in (F.linear, F.conv1d, F.conv2d):
            name, index = self._stack[-1] if self._stack else ("functional", 0)
            if self._stack:
                self._stack[-1] = (name, index + 1)
            key = f"{name}.{index}"
            with self._measurement():
                x = args[0] if args else kwargs["input"]
                w = args[1] if len(args) > 1 else kwargs["weight"]
                options = {}
                if func is not F.linear:
                    for i, (option, default) in enumerate(
                        (("stride", 1), ("padding", 0), ("dilation", 1), ("groups", 1)), 3
                    ):
                        options[option] = args[i] if len(args) > i else kwargs.get(option, default)
                owner = self._modules.get(name)
                if owner is not None and owner.__class__.__name__ == "QuantizeLinear" and owner.weight_bits == 1:
                    if not bool((x == x.round()).all()):
                        x = (x / owner.input_clip_val.clamp_min(1e-5)).round()
                    w = w.sign()
                    options.update(input_bits=owner.input_bits, weight_bits=1)
                self.observe(key, x, w, **options)
        return result

    def observe(
        self,
        key: str,
        x: Tensor,
        w: Tensor,
        *,
        stride: int | tuple[int, ...] = 1,
        padding: int | str | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        input_bits: int | None = None,
        weight_bits: int | None = None,
    ) -> None:
        input_bits = self.factory.input_bits if input_bits is None else input_bits
        weight_bits = self.factory.weight_bits if weight_bits is None else weight_bits
        x_code = integer_codes(x, input_bits)
        w_code = integer_codes(w, weight_bits)
        if w.ndim == 3:
            x_code = x_code.unsqueeze(-2)
            w_code = w_code.unsqueeze(-2)
            stride, padding, dilation = (1, _pair(stride)[0]), (0, _pair(padding)[0]), (1, _pair(dilation)[0])
        elif w.ndim == 4:
            stride, dilation = _pair(stride), _pair(dilation)
            if isinstance(padding, str):
                if padding == "same":
                    kh, kw = w.shape[-2:]
                    ph, pw = dilation[0] * (kh - 1), dilation[1] * (kw - 1)
                    x_code = F.pad(x_code, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
                padding = (0, 0)
            else:
                padding = _pair(padding)
        metric = self.measurements[key]
        metric.calls += 1
        for group in range(groups):
            gx = x_code if groups == 1 else x_code.chunk(groups, dim=-3)[group]
            gw = w_code if groups == 1 else w_code.chunk(groups, dim=0)[group]
            # Unsigned macros receive separate positive/negative weight planes.
            weights = (gw,) if self.factory.preset == "xue2020jssc" else (gw.clamp_min(0), (-gw).clamp_min(0))
            for wi, weight in enumerate(weights):
                unit_key = f"{key}.g{group}.w{wi}".replace(".", "_")
                if unit_key not in self.units:
                    device_ids = [x.device.index] if x.is_cuda else []
                    with torch.random.fork_rng(devices=device_ids):
                        unit = self.factory.build(
                            tuple(weight.shape),
                            stride=stride,
                            padding=(0, 0),
                            dilation=dilation,
                            input_bits=input_bits,
                            weight_bits=weight_bits,
                        )
                        unit.to(x.device).eval()
                        fabricate(unit)
                    self.units[unit_key] = unit
                    stamp_names(self.units)
                unit = self.units[unit_key]
                previous = self.weights.get(unit_key)
                if previous is None or not torch.equal(previous, weight):
                    unit.program(weight)
                    self.weights[unit_key] = weight.clone()
                operands = (gx,) if bool((gx >= 0).all()) else (gx.clamp_min(0), (-gx).clamp_min(0))
                for operand in operands:
                    if w.ndim == 2:
                        operand = operand.reshape(-1, operand.shape[-1])
                    else:
                        operand = operand.reshape(-1, *operand.shape[-3:])
                    for batch in operand.split(self.batch_chunk, dim=0):
                        chunks = (
                            (batch,)
                            if weight.ndim == 2
                            else convolution_chunks(
                                batch,
                                kernel=tuple(weight.shape[-2:]),
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                max_positions=self.spatial_chunk,
                            )
                        )
                        for chunk in chunks:
                            with Profiler() as profiler:
                                if weight.ndim == 2:
                                    unit.linear(chunk, quantization_mode=0, adc_active_bits=None)
                                else:
                                    unit.conv2d(chunk, quantization_mode=0, adc_active_bits=None)
                            metric.dynamic_energy__fJ += sum(
                                float(r.dynamic_energy__fJ.sum()) for r in profiler.records
                            )

    def report(self) -> dict:
        return {
            "preset": self.factory.preset,
            "ideal_macro": True,
            "analog_dynamic_energy_included": False,
            "quantization_defaults": {
                "input_bits": self.factory.input_bits,
                "weight_bits": self.factory.weight_bits,
                "floating_operands": "symmetric_per_tensor_absmax",
            },
            "merge": self.factory.merge,
            "digital_cost_assumptions__fJ": {
                "accumulator": self.factory.accumulator.energy_per_op__fJ,
                "shift_adder": self.factory.shift.energy_per_op__fJ,
            },
            "operator_calls": sum(m.calls for m in self.measurements.values()),
            "dynamic_energy__fJ": sum(m.dynamic_energy__fJ for m in self.measurements.values()),
            "operators": {name: vars(m) for name, m in self.measurements.items()},
        }
