"""Logical stimuli used by ADC-input characterization."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import ActiveRowSelection


def feasible_ideal_values(
    input_values: tuple[int, ...],
    weight_values: tuple[int, ...],
    active_num: int,
) -> tuple[int, ...]:
    """Return every dot product reachable by the declared value domains."""
    contributions = {input_ * weight for input_ in input_values for weight in weight_values}
    reachable = {0}
    for _ in range(active_num):
        reachable = {prefix + contribution for prefix in reachable for contribution in contributions}
    return tuple(sorted(reachable))


def sample_values(
    values: Tensor,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
) -> Tensor:
    index = torch.randint(values.numel(), shape, device=values.device, generator=generator)
    return values[index]


def sample_positions(
    *,
    batch_size: int,
    input_num: int,
    selected_num: int,
    selection: ActiveRowSelection,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    if selection is ActiveRowSelection.CONTIGUOUS:
        start = torch.randint(
            input_num - selected_num + 1,
            (batch_size, 1),
            device=device,
            generator=generator,
        )
        return start + torch.arange(selected_num, device=device).unsqueeze(0)
    return torch.rand((batch_size, input_num), device=device, generator=generator).topk(selected_num, dim=-1).indices


def sample_sparse_inputs(
    values: Tensor,
    *,
    batch_size: int,
    input_num: int,
    max_active_num: int,
    active_row_selection: ActiveRowSelection,
    generator: torch.Generator,
) -> Tensor:
    """Sample one legal input per batch element without a serial plane axis."""
    selected_num = min(max_active_num, input_num)
    selected_values = sample_values(values, (batch_size, selected_num), generator=generator)
    positions = sample_positions(
        batch_size=batch_size,
        input_num=input_num,
        selected_num=selected_num,
        selection=active_row_selection,
        device=values.device,
        generator=generator,
    )
    x = torch.zeros((batch_size, input_num), dtype=torch.int64, device=values.device)
    return x.scatter(1, positions, selected_values)


def _add_shifted(destination: Tensor, source: Tensor, shift: int) -> None:
    if shift >= 0:
        destination[shift:] += source[: source.numel() - shift]
    else:
        destination[:shift] += source[-shift:]


def _sum_probability_table(
    contributions: tuple[int, ...],
    *,
    item_num: int,
    lower: int,
    upper: int,
    device: torch.device,
) -> Tensor:
    width = upper - lower + 1
    offset = -lower
    table = torch.zeros((item_num + 1, width), dtype=torch.float64, device=device)
    table[0, offset] = 1.0
    for length in range(1, item_num + 1):
        for contribution in contributions:
            _add_shifted(table[length], table[length - 1], contribution)
        table[length] /= len(contributions)
    return table


def _lookup_probability(distribution: Tensor, sums: Tensor, *, offset: int) -> Tensor:
    indices = sums + offset
    valid = (indices >= 0) & (indices < distribution.shape[-1])
    indices = indices.clamp(0, distribution.shape[-1] - 1)
    if distribution.ndim == 1:
        return distribution[indices] * valid
    return torch.gather(distribution, 1, indices) * valid


def _add_row_shifts(destination: Tensor, source: Tensor, shifts: Tensor) -> None:
    width = source.shape[-1]
    source_indices = torch.arange(width, device=source.device).unsqueeze(0).expand_as(source)
    destination_indices = source_indices + shifts.unsqueeze(1)
    valid = (destination_indices >= 0) & (destination_indices < width)
    destination.scatter_add_(
        1,
        destination_indices.clamp(0, width - 1),
        source * valid,
    )


class TargetStimulusSampler:
    """Generate logical stimuli conditioned on one exact ideal dot product."""

    def __init__(
        self,
        *,
        input_values: tuple[int, ...],
        weight_values: tuple[int, ...],
        active_num: int,
        input_num: int,
        output_num: int,
        target_ideal_values: tuple[int, ...],
        active_row_selection: ActiveRowSelection,
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self._input_values = torch.tensor(input_values, dtype=torch.int64, device=device)
        self._weight_values = torch.tensor(weight_values, dtype=torch.int64, device=device)
        self._active_num = active_num
        self._input_num = input_num
        self._output_num = output_num
        self._active_row_selection = active_row_selection
        self._device = device
        self._generator = generator

        pair_inputs = tuple(input_ for input_ in input_values for _ in weight_values)
        pair_weights = tuple(weight for _ in input_values for weight in weight_values)
        pair_contributions = tuple(input_ * weight for input_, weight in zip(pair_inputs, pair_weights, strict=True))
        self._pair_inputs = torch.tensor(pair_inputs, dtype=torch.int64, device=device)
        self._pair_contributions = torch.tensor(pair_contributions, dtype=torch.int64, device=device)

        min_contribution = min(pair_contributions)
        max_contribution = max(pair_contributions)
        self._lower = min(0, active_num * min_contribution)
        self._upper = max(0, active_num * max_contribution)
        self._offset = -self._lower
        self._pair_probability = _sum_probability_table(
            pair_contributions,
            item_num=active_num,
            lower=self._lower,
            upper=self._upper,
            device=device,
        )
        for target in target_ideal_values:
            if not self._lower <= target <= self._upper:
                raise ValueError(
                    f"require: target ideal value ({target}) in feasible support [{self._lower}, {self._upper}]"
                )
            if not bool(self._pair_probability[active_num, target + self._offset] > 0.0):
                raise ValueError(f"require: target ideal value ({target}) has at least one input/weight realization")

    def _sample_input_values(self, target: int, batch_size: int) -> Tensor:
        remainder = torch.full((batch_size,), target, dtype=torch.int64, device=self._device)
        selected = torch.empty((batch_size, self._active_num), dtype=torch.int64, device=self._device)
        for index in range(self._active_num):
            remaining_num = self._active_num - index - 1
            suffix_sums = remainder.unsqueeze(1) - self._pair_contributions.unsqueeze(0)
            probability = _lookup_probability(
                self._pair_probability[remaining_num],
                suffix_sums,
                offset=self._offset,
            )
            choice = torch.multinomial(probability, 1, generator=self._generator).squeeze(1)
            selected[:, index] = self._pair_inputs[choice]
            remainder -= self._pair_contributions[choice]
        return selected

    def _weight_suffix_probability(self, selected_inputs: Tensor) -> Tensor:
        batch_size = selected_inputs.shape[0]
        width = self._upper - self._lower + 1
        suffix = torch.zeros(
            (self._active_num + 1, batch_size, width),
            dtype=torch.float64,
            device=self._device,
        )
        suffix[self._active_num, :, self._offset] = 1.0
        for index in range(self._active_num - 1, -1, -1):
            for weight in self._weight_values:
                _add_row_shifts(
                    suffix[index],
                    suffix[index + 1],
                    selected_inputs[:, index] * weight,
                )
            suffix[index] /= self._weight_values.numel()
        return suffix

    def _sample_active_weights(self, selected_inputs: Tensor, target: int) -> Tensor:
        batch_size = selected_inputs.shape[0]
        suffix = self._weight_suffix_probability(selected_inputs)
        remainder = torch.full(
            (batch_size, self._output_num),
            target,
            dtype=torch.int64,
            device=self._device,
        )
        selected = torch.empty(
            (batch_size, self._active_num, self._output_num),
            dtype=torch.int64,
            device=self._device,
        )
        for index in range(self._active_num):
            probability = torch.stack(
                [
                    _lookup_probability(
                        suffix[index + 1],
                        remainder - selected_inputs[:, index].unsqueeze(1) * weight,
                        offset=self._offset,
                    )
                    for weight in self._weight_values
                ],
                dim=-1,
            )
            choice = torch.multinomial(
                probability.reshape(-1, self._weight_values.numel()),
                1,
                generator=self._generator,
            ).reshape(batch_size, self._output_num)
            weight = self._weight_values[choice]
            selected[:, index] = weight
            remainder -= selected_inputs[:, index].unsqueeze(1) * weight
        return selected

    def sample(self, target: int, batch_size: int) -> tuple[Tensor, Tensor]:
        """Generate one fully materialized weight/input batch on the configured device."""
        selected_inputs = self._sample_input_values(target, batch_size)
        positions = sample_positions(
            batch_size=batch_size,
            input_num=self._input_num,
            selected_num=self._active_num,
            selection=self._active_row_selection,
            device=self._device,
            generator=self._generator,
        )
        x = torch.zeros((batch_size, self._input_num), dtype=torch.int64, device=self._device)
        x.scatter_(1, positions, selected_inputs)

        selected_weights = self._sample_active_weights(selected_inputs, target)
        w = sample_values(
            self._weight_values,
            (batch_size, self._input_num, self._output_num),
            generator=self._generator,
        )
        w.scatter_(
            1,
            positions.unsqueeze(-1).expand(batch_size, self._active_num, self._output_num),
            selected_weights,
        )
        return w, x.unsqueeze(0)
