"""P1 probe — verify a module-level free function decorated with
``@torch.compile(dynamic=False)`` shares its compiled graph across
*different* module instances when input *shape* is identical.

This is the single most load-bearing assumption of the chunking plan:
if torch.compile keys on ``id(self)``-flavoured state (different RRAM
buffer instance ⇒ different trace), no amount of canonical-shape work
buys us anything — we'd still cold-compile every macro.

Setup mirrors the real solver hot path at a smaller scale:

  * ``_MockCore`` is a tiny nn.Module holding two registered buffers
    (mimicking ``rram.g__uS`` and ``nmos.vth__V``). Two instances are
    built with different random fab states but the same shape.
  * ``_solve_chunk_free`` is a module-level function that takes the two
    buffers + an input tensor + an outer "snapshot" tensor, does a
    little arithmetic, and returns ``(v_out, energy)``. It is decorated
    with ``@torch.compile(dynamic=False)``.
  * Cold call: instance 0 invokes the free function for the first time.
  * Cross-instance call: instance 1 invokes the free function with the
    same input shape but its own buffers.

If torch.compile correctly keys only on the function code + input
shapes/dtypes, the cross-instance call should be ~0s (warm cache hit).
If it keys on ``self`` or on the buffer identity, the cross-instance
call will trigger another full cold compile.
"""

# ruff: noqa: T201

import time

import torch
import torch.nn as nn
from torch import Tensor


@torch.compile(dynamic=False)
def _solve_chunk_free(
    x: Tensor,
    g_static: Tensor,
    vth: Tensor,
    snap_noise: Tensor,
) -> tuple[Tensor, Tensor]:
    v_drive = torch.tanh(x) + snap_noise
    i_cell = g_static * (v_drive - vth)
    v_out = i_cell.sum(dim=-1)
    energy = (v_drive * i_cell).sum(dim=(-2, -1)).abs()
    return v_out, energy


class _MockCore(nn.Module):
    """Tiny stand-in for ``Core1T1R`` carrying two fab buffers."""

    g_static: Tensor
    vth: Tensor

    def __init__(self, inst_size: int, tc: int, col: int, row: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        gen.manual_seed(seed)
        self.register_buffer(
            "g_static",
            torch.rand((inst_size, tc, col, row), generator=gen, device=gen.device),
            persistent=False,
        )
        self.register_buffer(
            "vth",
            torch.rand((inst_size, tc, col, row), generator=gen, device=gen.device),
            persistent=False,
        )

    def solve_output(self, x: Tensor, snap_noise: Tensor) -> tuple[Tensor, Tensor]:
        return _solve_chunk_free(x, self.g_static, self.vth, snap_noise)


def _time_solve(core: _MockCore, x: Tensor, snap: Tensor, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    out = core.solve_output(x, snap)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time() - t0, out


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[p1] device={device}")

    inst_size = 8
    tc = 4
    col = 16
    row = 16
    chunk_x = 8

    core0 = _MockCore(inst_size, tc, col, row, seed=0).to(device).eval()
    core1 = _MockCore(inst_size, tc, col, row, seed=1).to(device).eval()
    core2_diff_shape = _MockCore(inst_size, tc, col, row * 2, seed=2).to(device).eval()
    core3_revisit = _MockCore(inst_size, tc, col, row, seed=3).to(device).eval()

    shape_x = (chunk_x, 1, tc, 1, row)
    shape_snap = (chunk_x, inst_size, tc, col, row)
    shape_x_diff = (chunk_x, 1, tc, 1, row * 2)
    shape_snap_diff = (chunk_x, inst_size, tc, col, row * 2)

    x0 = torch.randn(shape_x, device=device)
    snap0 = torch.randn(shape_snap, device=device)
    x1 = torch.randn(shape_x, device=device)
    snap1 = torch.randn(shape_snap, device=device)
    x2 = torch.randn(shape_x_diff, device=device)
    snap2 = torch.randn(shape_snap_diff, device=device)
    x3 = torch.randn(shape_x, device=device)
    snap3 = torch.randn(shape_snap, device=device)

    t1, _ = _time_solve(core0, x0, snap0, device)
    print(f"[p1] call 1 (cold, core0): {t1:.3f}s")

    t2, _ = _time_solve(core1, x1, snap1, device)
    print(f"[p1] call 2 (cross-instance core1, same shape): {t2:.4f}s")

    t3, _ = _time_solve(core2_diff_shape, x2, snap2, device)
    print(f"[p1] call 3 (different shape core2): {t3:.3f}s  (expect cold compile)")

    t4, _ = _time_solve(core3_revisit, x3, snap3, device)
    print(f"[p1] call 4 (cross-instance core3, original shape revisit): {t4:.4f}s")

    print("[p1] -----------------------------------------------")
    share_ratio = t2 / t1 if t1 > 0 else 0.0
    revisit_ratio = t4 / t1 if t1 > 0 else 0.0
    print(f"[p1] cross-instance/cold ratio (call 2 / call 1): {share_ratio:.3%}  (want <10%)")
    print(f"[p1] re-visit/cold ratio (call 4 / call 1): {revisit_ratio:.3%}  (want <10%)")
    print(f"[p1] different-shape cold (call 3): {t3:.3f}s")

    if share_ratio < 0.10 and revisit_ratio < 0.10:
        print("[p1] PASS — free function shares compile cache across module instances")
    else:
        print("[p1] FAIL — cross-instance not sharing; revisit free-function strategy")


if __name__ == "__main__":
    main()
