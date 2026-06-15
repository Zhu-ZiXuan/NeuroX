"""Capture-only ADC stand-in for ``statistic_xbar_adc``.

See also:
    docs/modules/tools/xbar_adc/statistic.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType

import torch
import torch.nn as nn
from torch import Tensor

from neurox.analog.adc import ADC, ADCConfig, AdcOperationPoint, ADCPolicy
from neurox.xbar import Offset1T1RXbar
from neurox.xbar.readout import OffsetSwitchCapMuxAdcReadOut


class ProbeADC(ADC):
    """Capture-only ADC stand-in for analog-input distribution probing.

    Tool-local test double. Intentionally NOT registered with the ADC
    family registry (no ``@ADC.register_key``) and NOT exported from
    ``neurox/analog/adc/``. The class only exists to satisfy the
    readout chain's static ADC interface so that
    :meth:`OffsetSwitchCapMuxAdcReadOut.readout` can call
    ``bl_adc.convert(v_pos__V, v_neg__V, ...)`` without a type error,
    while letting the tool capture the analog ADC input that would
    otherwise be lost to quantization.

    Args:
        mode_num: ``mode_num`` of the replaced real ADC.
        max_bits: ``max_bits`` of the replaced real ADC.
        signed_range_callable: The real ADC's bound ``signed_range``
            method; called verbatim so downstream saturation tests
            (e.g. ``calibrate.saturation_mask``) get the actual ADC's
            realised endpoints rather than the canonical SAR bounds.
        name: Hierarchical instance name (typically the replaced ADC's
            ``qualified_name``).
        inst_shape: The replaced ADC's instance shape.
    """

    def __init__(
        self,
        *,
        mode_num: int,
        max_bits: int,
        signed_range_callable: Callable[[int], tuple[int, int]],
        name: str,
        inst_shape: tuple[int, ...],
    ) -> None:
        # ADCConfig inherits CircuitConfig (area / leakage); pass zeros
        # for the dummy probe instance. ADC.__init__ then deletes
        # policy/dtype/T__K immediately. ProbeADC never logs dynamic
        # events so per-op latency is irrelevant.
        super().__init__(
            config=ADCConfig(area_per_inst__um2=0.0, leakage_per_inst__uW=0.0),
            policy=ADCPolicy(),
            name=name,
            inst_shape=inst_shape,
            dtype=torch.float32,
            T__K=0.0,
        )
        self._mode_num = mode_num
        self._max_bits = max_bits
        self._signed_range_callable = signed_range_callable
        self._buf_pos: list[Tensor] = []
        self._buf_neg: list[Tensor] = []

    @property
    def mode_num(self) -> int:
        return self._mode_num

    @property
    def max_bits(self) -> int:
        return self._max_bits

    def signed_range(self, adc_bits: int) -> tuple[int, int]:
        """Delegate to the replaced ADC's realised signed range."""
        return self._signed_range_callable(adc_bits)

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Capture v_pos / v_neg, return a dummy zero code.

        The dummy code keeps the downstream
        :meth:`OffsetSwitchCapMuxAdcReadOut.readout` flatten chain valid;
        its value is ignored by the tool.
        """
        del adc_operation_point
        self._buf_pos.append(v_pos__V.detach().cpu().flatten().to(torch.float64))
        self._buf_neg.append(v_neg__V.detach().cpu().flatten().to(torch.float64))
        return torch.zeros_like(v_pos__V, dtype=torch.int64)

    def captured(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return concatenated ``(v_pos, v_neg, v_diff)`` — CPU float64 1-D."""
        if not self._buf_pos:
            empty = torch.empty(0, dtype=torch.float64)
            return empty, empty.clone(), empty.clone()
        pos = torch.cat(self._buf_pos)
        neg = torch.cat(self._buf_neg)
        return pos, neg, pos - neg


@dataclass
class ProbeHandle:
    """Reversible ``bl_adc`` replacement record.

    Stores ``original_adc`` outside the :class:`ProbeADC` module tree
    so that ``physical.modules()`` traversal does not see the original
    ADC twice while the probe is installed.

    Use as a context manager — :meth:`restore` runs on exit.

    Attributes:
        owner: The module whose attribute is being swapped.
        attr: The attribute name on ``owner`` (``"bl_adc"``).
        original_adc: The replaced ADC module — kept here, not inside
            the probe, to avoid module-tree duplication.
        probe: The installed :class:`ProbeADC`.
    """

    owner: nn.Module
    attr: str
    original_adc: nn.Module
    probe: ProbeADC

    def __enter__(self) -> ProbeHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.restore()

    def restore(self) -> None:
        """Reinstall the original ADC on the owner."""
        setattr(self.owner, self.attr, self.original_adc)


def install_probe_adc(xbar: Offset1T1RXbar) -> ProbeHandle:
    """Replace ``xbar.readout.bl_adc`` with a :class:`ProbeADC`.

    Topology-aware: only :class:`OffsetSwitchCapMuxAdcReadOut` is
    supported. Recursive ``named_modules()`` scanning is deliberately
    avoided so that a future readout carrying multiple ADCs cannot
    silently mis-target the probe.

    Args:
        xbar: An :class:`Offset1T1RXbar` whose readout will be probed.

    Returns:
        A :class:`ProbeHandle` whose ``restore()`` reinstalls the
        original ADC.

    Raises:
        TypeError: When the xbar's readout is a different topology.
    """
    readout = xbar.readout
    if not isinstance(readout, OffsetSwitchCapMuxAdcReadOut):
        raise TypeError(
            f"install_probe_adc: unsupported readout type {type(readout).__name__}; "
            "only OffsetSwitchCapMuxAdcReadOut is supported."
        )
    real_adc = readout.bl_adc
    probe = ProbeADC(
        mode_num=real_adc.mode_num,
        max_bits=real_adc.max_bits,
        signed_range_callable=real_adc.signed_range,
        name=real_adc.qualified_name,
        inst_shape=real_adc.inst_shape,
    )
    readout.bl_adc = probe
    return ProbeHandle(owner=readout, attr="bl_adc", original_adc=real_adc, probe=probe)
