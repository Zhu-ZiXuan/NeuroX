"""P2 probe — measure peak GPU memory and compile time of a single
canonical-shape solver chunk for BERT-FFN-scale workloads.

The chunk shape mirrors the post-flatten canonical layout:
``[chunk_x, B_inst, Tc, col, row]`` where every axis is a policy or
chip-level constant. The probe sweeps a few realistic combinations:

  * LeNet-scale: chunk_x=16, B_inst=1, Tc=1, col=row=64
  * BERT-QKV:    chunk_x=16, B_inst=48, Tc=12, col=row=64
  * BERT-FFN:    chunk_x=16, B_inst=192, Tc=12, col=row=64

For each we materialise a handful of ``[chunk_x, B_inst, Tc, col, row]``
node-voltage-sized tensors (mimicking the DCOP footprint), run a couple
of fake Newton steps, and report peak ``torch.cuda.max_memory_allocated``.

A ``@torch.compile(dynamic=False)`` decorator captures cold compile
time. Goal: peak per-chunk memory under 1 GB for BERT-FFN, demonstrating
the chunking plan actually fixes the OOM problem.
"""

# ruff: noqa: T201

import time

import torch
from torch import Tensor


@torch.compile(dynamic=False)
def _mock_solver_chunk(
    x: Tensor,             # [chunk_x, 1, Tc, 1, row]
    g_static: Tensor,      # [1, B_inst, Tc, col, row]
    rram_g_snap: Tensor,   # [chunk_x, B_inst, Tc, col, row]
    nmos_snap: Tensor,     # [chunk_x, B_inst, Tc, col, row]
) -> tuple[Tensor, Tensor]:
    """Mimic the broadcast + a few KCL-style passes that the real Newton
    solver does on canonical-shape inputs. The arithmetic doesn't have
    to be physically meaningful — only the tensor footprints do.
    """
    g = g_static * rram_g_snap

    v_drive = torch.tanh(x).expand_as(g)
    v_bl = v_drive * 0.5
    v_sl = v_drive * 0.1
    v_x = v_drive * 0.05
    i_cell = g * (v_bl - v_x) - nmos_snap

    for _ in range(3):
        delta = (v_bl - v_sl - v_x) * 0.01
        v_bl = v_bl - delta
        v_sl = v_sl + delta * 0.5
        v_x = v_x + delta * 0.2
        i_cell = g * (v_bl - v_x) - nmos_snap

    v_out_phys = i_cell.sum(dim=-1).sum(dim=-1)
    energy = (v_bl * i_cell + v_sl * i_cell).sum(dim=(-2, -1)).abs()
    return v_out_phys, energy


def _peak_mem_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _run_case(label: str, chunk_x: int, b_inst: int, tc: int, col: int, row: int) -> None:
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn((chunk_x, 1, tc, 1, row), device=device)
    g_static = torch.rand((1, b_inst, tc, col, row), device=device) * 0.5 + 0.5
    rram_g_snap = torch.rand((chunk_x, b_inst, tc, col, row), device=device) * 0.1 + 1.0
    nmos_snap = torch.randn((chunk_x, b_inst, tc, col, row), device=device) * 0.001

    free_before, _ = torch.cuda.mem_get_info(device)
    print(f"[p2:{label}] device free before: {free_before/1e9:.2f} GB")
    print(f"[p2:{label}] inputs: x={tuple(x.shape)}  g_static={tuple(g_static.shape)}  snap={tuple(rram_g_snap.shape)}")

    torch.cuda.synchronize(device)
    t0 = time.time()
    v_out, e = _mock_solver_chunk(x, g_static, rram_g_snap, nmos_snap)
    torch.cuda.synchronize(device)
    cold_t = time.time() - t0
    cold_peak = _peak_mem_gb(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(device)
    t1 = time.time()
    v_out, e = _mock_solver_chunk(x, g_static, rram_g_snap, nmos_snap)
    torch.cuda.synchronize(device)
    warm_t = time.time() - t1
    warm_peak = _peak_mem_gb(device)

    print(
        f"[p2:{label}] cold={cold_t:.2f}s  warm={warm_t:.4f}s  "
        f"cold_peak={cold_peak:.3f} GB  warm_peak={warm_peak:.3f} GB  "
        f"v_out_sum={v_out.sum().item():.2e}"
    )

    # Clean for next case.
    del x, g_static, rram_g_snap, nmos_snap, v_out, e
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        print("[p2] no CUDA device — skipping memory probe")
        return

    print("[p2] sweeping canonical chunk shapes (real per-chunk footprints)")
    print("[p2] ---------------------------------------------------------")

    # NOTE: same compiled function is reused across cases; shape changes
    # trigger recompile, so cold times are case-specific.
    _run_case("lenet",     chunk_x=16, b_inst=1,   tc=1,  col=64, row=64)
    _run_case("bert-qkv",  chunk_x=16, b_inst=48,  tc=12, col=64, row=64)
    _run_case("bert-ffn",  chunk_x=16, b_inst=192, tc=12, col=64, row=64)


if __name__ == "__main__":
    main()
