"""Tests for fixed-shape solver chunk construction."""

import torch

from neurox.primitive.xbar.solver import iter_chunks, reassemble_chunks


def test_positive_chunk_size_pads_only_solver_coordinates() -> None:
    specs = list(iter_chunks(leading=(5,), chunk_size=3, device=torch.device("cpu")))

    assert [spec.solve_size for spec in specs] == [3, 3]
    assert [spec.valid_size for spec in specs] == [3, 2]
    torch.testing.assert_close(specs[0].multi_coords[0], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(specs[1].multi_coords[0], torch.tensor([3, 4, 4]))
    torch.testing.assert_close(specs[1].flat_global_idx, torch.tensor([3, 4]))


def test_reassembly_omits_padded_solver_positions() -> None:
    specs = list(iter_chunks(leading=(5,), chunk_size=3, device=torch.device("cpu")))
    solved = [spec.multi_coords[0][: spec.valid_size] * 10 for spec in specs]

    actual = reassemble_chunks(
        solved,
        [spec.flat_global_idx for spec in specs],
        leading=(5,),
        trailing=(),
    )

    torch.testing.assert_close(actual, torch.tensor([0, 10, 20, 30, 40]))


def test_non_positive_chunk_size_keeps_one_unpadded_chunk() -> None:
    [spec] = list(iter_chunks(leading=(2, 3), chunk_size=0, device=torch.device("cpu")))

    assert spec.solve_size == 6
    assert spec.valid_size == 6
    torch.testing.assert_close(spec.flat_global_idx, torch.arange(6))


def test_chunk_larger_than_workload_still_uses_fixed_solver_size() -> None:
    [spec] = list(iter_chunks(leading=(2,), chunk_size=5, device=torch.device("cpu")))

    assert spec.solve_size == 5
    assert spec.valid_size == 2
    torch.testing.assert_close(spec.multi_coords[0], torch.tensor([0, 1, 1, 1, 1]))


def test_empty_leading_shape_remains_atomic_and_unpadded() -> None:
    [spec] = list(iter_chunks(leading=(), chunk_size=5, device=torch.device("cpu")))

    assert spec.multi_coords == ()
    assert spec.solve_size == 1
    assert spec.valid_size == 1
