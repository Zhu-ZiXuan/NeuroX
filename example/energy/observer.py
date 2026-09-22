"""Bind fixed hardware to complete operators and observe their inputs only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

from neurox import Profiler, fabricate
from neurox.architecture.unit.conv2d import Conv2dCimUnit
from neurox.architecture.unit.linear import LinearCimUnit

from .factory import UnitFactory


def integer_codes(value: Tensor, bits: int) -> Tensor:
    """Require existing integer levels within the declared magnitude precision."""
    value = value.detach()
    limit = (1 << bits) - 1
    rounded = value.round()
    if not bool(((value == rounded) & (value.abs() <= limit)).all()):
        raise ValueError(f"operand has no integer codes within the declared {bits}-bit magnitude range")
    return rounded.to(torch.int32)


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    first, second = value
    return first, second


class UnsupportedLayersError(ValueError):
    """Carry unsupported complete-layer names and their rejection reasons."""

    def __init__(self, layers: dict[str, str]) -> None:
        self.layers = layers
        super().__init__("Unsupported NeuroX layers:\n" + "\n".join(f"  {name}: {why}" for name, why in layers.items()))


@dataclass(kw_only=True)
class _Operator:
    owner: nn.Module
    attribute: str
    source: str
    input_bits: int
    weight_bits: int
    input_mode: str
    input_rank: int
    convolution: bool
    conv1d: bool
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_padding: tuple[int, int, int, int]
    padding_mode: str
    unit: LinearCimUnit | Conv2dCimUnit | None = None
    leading_shapes: list[tuple[int, ...]] = field(default_factory=list)


class EnergyObserver(TorchFunctionMode):
    """Keep model outputs while measuring each complete operator on its own unit.

    `prepare` runs one discovery forward outside profiling, restores model
    state and RNG, and programs a separate unit for every observed operator
    position. Units are registered on their owning model modules. Evaluation
    keeps that topology and those programmed weights fixed.

    Ordinary functional calls are observed before their operands are tiled.
    SOUL QLinear/QConv modules are observed at their complete-layer input;
    their internal tiled implementation remains solely the numerical reference.
    The caller owns one Profiler context per model batch and reads its results.
    """

    def __init__(self, model: nn.Module, factory: UnitFactory) -> None:
        super().__init__()
        self.model = model
        self.factory = factory
        self.operators: dict[tuple[str, int], _Operator] = {}
        self._modules = dict(model.named_modules())
        self._weights: dict[tuple[str, int], Tensor] = {}
        self._unsupported: dict[str, str] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._stack: list[str] = []
        self._hooks: list[RemovableHandle] = []
        self._preparing = False
        self._busy = False
        self._handled_depth = 0
        self._ready = False

    def __enter__(self) -> Self:
        for name, module in self._modules.items():
            self._hooks.append(module.register_forward_pre_hook(self._before(name), with_kwargs=True))
            self._hooks.append(module.register_forward_hook(self._after, with_kwargs=True, always_call=True))
        super().__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
            self._stack.clear()
            self._handled_depth = 0

    # === Public API ===

    def prepare(self, operands: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Discover and bind operators once, before entering any Profiler context."""
        if self._ready or Profiler.active():
            raise RuntimeError("prepare once, before entering any Profiler context")
        state = {name: value.clone() for name, value in self.model.state_dict().items()}
        devices = sorted({p.device.index for p in self.model.parameters() if p.is_cuda})
        self._preparing = True
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=devices), self:
                self.model(*operands, **kwargs)
        finally:
            self._preparing = False
            self.model.load_state_dict(state)
        if self._unsupported:
            self._weights.clear()
            raise UnsupportedLayersError(self._unsupported)
        if not self.operators:
            raise ValueError("model contains no observed linear or convolution operators")

        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            for key, operator in self.operators.items():
                weight = self._weights.pop(key)
                unit: LinearCimUnit | Conv2dCimUnit
                if operator.convolution:
                    unit = self.factory.build_conv2d(
                        tuple(weight.shape),
                        stride=operator.stride,
                        padding=operator.padding,
                        dilation=operator.dilation,
                        groups=operator.groups,
                        input_bits=operator.input_bits,
                        weight_bits=operator.weight_bits,
                    )
                else:
                    unit = self.factory.build_linear(
                        tuple(weight.shape), input_bits=operator.input_bits, weight_bits=operator.weight_bits
                    )
                unit.to(weight.device).eval()
                fabricate(unit)
                unit.program(weight)
                unit.set_profile_leading_rank(1 if operator.convolution else max(1, operator.input_rank - 1))
                operator.owner.add_module(operator.attribute, unit)
                operator.unit = unit
        self._ready = True

    # === Input observation ===

    @contextmanager
    def _measurement(self) -> Iterator[None]:
        self._busy = True
        try:
            with torch.no_grad():
                yield
        finally:
            self._busy = False

    def _before(self, name: str) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any]], None]:
        def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if not self._stack:
                self._counts.clear()
            self._stack.append(name)
            if module.__class__.__name__ in ("QLinear", "QConv") and hasattr(module, "is_spike"):
                self._handled_depth += 1
                if self._preparing or Profiler.active():
                    with self._measurement():
                        layer: Any = module
                        x = args[0] if args else kwargs["x"]
                        spike = module.mode == "adc" and module.is_spike and module.sparse
                        dense_q8 = module.mode == "adc" and not spike and module.dense_mode == "q8"
                        input_mode = "spike" if spike else "dense_q8" if dense_q8 else "floating"
                        options: dict[str, Any] = {
                            "input_bits": 1 if spike else self.factory.input_bits,
                            "weight_bits": self.factory.weight_bits if module.mode == "fp" else 1,
                            "input_mode": input_mode,
                        }
                        if dense_q8:
                            options["input_bits"] = module.act_bits
                        if module.__class__.__name__ == "QLinear":
                            weight: Tensor = layer.weight
                            if self._preparing and module.mode != "fp":
                                weight = weight.sign()
                            self._observe(name, x, weight, **options)
                        else:
                            conv: nn.Conv1d | nn.Conv2d = layer.conv
                            weight = conv.weight
                            if self._preparing and module.mode != "fp":
                                weight = weight.sign()
                            self._observe(
                                name,
                                x,
                                weight,
                                stride=conv.stride,
                                padding=conv.padding,
                                dilation=conv.dilation,
                                groups=conv.groups,
                                padding_mode=conv.padding_mode,
                                **options,
                            )

        return hook

    def _after(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: object) -> None:
        if module.__class__.__name__ in ("QLinear", "QConv") and hasattr(module, "is_spike"):
            self._handled_depth -= 1
        self._stack.pop()

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        kwargs = {} if kwargs is None else kwargs
        result = func(*args, **kwargs)
        if (
            self._busy
            or self._handled_depth
            or not (self._preparing or Profiler.active())
            or func not in (F.linear, F.conv1d, F.conv2d)
        ):
            return result
        name = self._stack[-1] if self._stack else ""
        with self._measurement():
            x = args[0] if args else kwargs["input"]
            weight = args[1] if len(args) > 1 else kwargs["weight"]
            options: dict[str, Any] = {}
            if func is not F.linear:
                for index, (option, default) in enumerate(
                    (("stride", 1), ("padding", 0), ("dilation", 1), ("groups", 1)), 3
                ):
                    options[option] = args[index] if len(args) > index else kwargs.get(option, default)
            owner = self._modules[name]
            if owner.__class__.__name__ == "QuantizeLinear" and owner.input_bits < 32 and owner.weight_bits == 1:
                if self._preparing:
                    weight = weight.sign()
                options.update(input_bits=owner.input_bits, weight_bits=1, input_mode="quantized_linear")
            self._observe(name, x, weight, **options)
        return result

    def _encode_input(self, operator: _Operator, x: Tensor) -> Tensor:
        owner: Any = operator.owner
        if operator.input_mode == "floating":
            raise ValueError("floating activations have no integer quantization in this layer's current mode")
        if operator.input_mode == "spike":
            alpha: Tensor = owner.act_alpha
            x = (x / alpha.clamp_min(1e-8)).round().clamp(0, 1)
        elif operator.input_mode == "dense_q8":
            limit = (1 << (operator.input_bits - 1)) - 1
            alpha = owner.act_alpha
            step = alpha.clamp_min(1e-8) / limit
            x = (x / step).round().clamp(-limit - 1, limit)
        elif operator.input_mode == "quantized_linear" and self.factory.quantized_linear_input == "scaled":
            alpha = owner.input_clip_val
            x = (x / alpha.clamp_min(1e-5)).round()
        code = integer_codes(x, operator.input_bits)
        if operator.convolution:
            if operator.conv1d:
                code = code.unsqueeze(-1)
            if any(operator.input_padding):
                code = F.pad(code, operator.input_padding, mode=operator.padding_mode)
        return code

    def _observe(
        self,
        name: str,
        x: Tensor,
        weight: Tensor,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | str | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        input_bits: int | None = None,
        weight_bits: int | None = None,
        input_mode: str = "generic",
    ) -> None:
        index = self._counts[name]
        self._counts[name] += 1
        key = name, index
        if self._preparing:
            convolution = weight.ndim != 2
            conv1d = weight.ndim == 3
            stride_hw = (stride if isinstance(stride, int) else stride[0], 1) if conv1d else _pair(stride)
            dilation_hw = (dilation if isinstance(dilation, int) else dilation[0], 1) if conv1d else _pair(dilation)
            if conv1d:
                weight = weight.unsqueeze(-1)
            if isinstance(padding, str):
                ph, pw = (d * (k - 1) for d, k in zip(dilation_hw, weight.shape[-2:], strict=True))
                input_padding = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2) if padding == "same" else (0, 0, 0, 0)
                padding_hw = (0, 0)
            else:
                padding_hw = (padding if isinstance(padding, int) else padding[0], 0) if conv1d else _pair(padding)
                ph, pw = padding_hw
                input_padding = (pw, pw, ph, ph) if padding_mode != "zeros" else (0, 0, 0, 0)
                if padding_mode != "zeros":
                    padding_hw = (0, 0)
            operator = _Operator(
                owner=self._modules[name],
                attribute=f"neurox_unit_{index}",
                source=f"{name or '<root>'}:{index}:{'conv1d' if conv1d else 'conv2d' if convolution else 'linear'}",
                input_bits=self.factory.input_bits if input_bits is None else input_bits,
                weight_bits=self.factory.weight_bits if weight_bits is None else weight_bits,
                input_mode=input_mode,
                input_rank=x.ndim,
                convolution=convolution,
                conv1d=conv1d,
                stride=stride_hw,
                padding=padding_hw,
                dilation=dilation_hw,
                groups=groups,
                input_padding=input_padding,
                padding_mode="constant" if padding_mode == "zeros" else padding_mode,
            )
            self.operators[key] = operator
            reasons = []
            if not self.factory.supports_precision(input_bits=operator.input_bits, weight_bits=operator.weight_bits):
                reasons.append(f"no local unit configuration for W{operator.weight_bits}A{operator.input_bits}")
            try:
                self._weights[key] = integer_codes(weight, operator.weight_bits).clone()
            except ValueError as error:
                reasons.append(f"weight: {error}")
            try:
                x_code = self._encode_input(operator, x)
                if bool((x_code < 0).any()):
                    reasons.append("signed inputs require an unsupported mapping onto this macro's unsigned input")
            except ValueError as error:
                reasons.append(f"input: {error}")
            if reasons:
                self._unsupported[operator.source] = "; ".join(reasons)
            return

        if not self._ready:
            raise RuntimeError("prepare the observer before profiling")
        operator = self.operators[key]
        unit = operator.unit
        if unit is None:
            raise RuntimeError(f"{operator.source} has no prepared unit")
        try:
            code = self._encode_input(operator, x)
        except ValueError as error:
            raise UnsupportedLayersError({operator.source: str(error)}) from error
        if bool((code < 0).any()):
            raise UnsupportedLayersError({operator.source: "signed input encountered after preparation"})
        leading_shape = tuple(code.shape[:-3] if operator.convolution else code.shape[:-1])
        devices = [code.device.index] if code.is_cuda else []
        stance = torch.compiler.set_stance("force_eager") if self.factory.ideal_macro else nullcontext()
        with torch.random.fork_rng(devices=devices), stance:
            if isinstance(unit, Conv2dCimUnit):
                unit.conv2d(code.reshape(-1, *code.shape[-3:]), quantization_mode=0, adc_active_bits=None)
            else:
                # A single vector still occupies one sample position across contexts.
                unit.linear(code.unsqueeze(0) if code.ndim == 1 else code, quantization_mode=0, adc_active_bits=None)
        operator.leading_shapes.append(leading_shape or (1,))
