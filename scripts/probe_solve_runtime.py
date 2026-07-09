"""Decisive: does compiling the DC solve actually make it FASTER than the
original eager Thomas solve at BERT-FFN scale? The bert eval OOM'd at the
eager chunk size and timed out at the memory-safe one, suggesting the
compiled PCR solve is both heavier and slower than eager Thomas.

Times one ``cim_read`` (a single chunk's worth of leading) three ways at a
representative FFN-chunk leading, reporting warm wall + peak memory:

  * eager-thomas   -- the ORIGINAL backend, eager (what scheme A replaced)
  * eager-pcr      -- PCR backend, eager (isolates PCR's extra work)
  * compiled-pcr   -- PCR backend, solve_dc compiled (scheme A)

Run: CUDA_VISIBLE_DEVICES=0 python scripts/probe_solve_runtime.py --leading 16384
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import neurox.primitive.xbar.solver.nested as nested_solver  # noqa: E402
from neurox.primitive.xbar.solver._linalg import solve_block_tridiagonal, solve_block_tridiagonal_pcr  # noqa: E402
from works.offset_1t1r.tools.macro_adc._sampling import build_offset_1t1r_xbar_all_off  # noqa: E402

XBAR_CONFIG = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"


def _build(device, dtype, leading):
    xbar = build_offset_1t1r_xbar_all_off(XBAR_CONFIG, device=device, dtype=dtype, solve_chunk_size=0)
    lo, hi = xbar.w_digit_range
    g = torch.Generator(device=device).manual_seed(0)
    xbar.program(torch.randint(lo, hi + 1, xbar._w_layout_shape, generator=g, device=device))
    row = xbar.core.fabricated_row_num
    xl, xh = xbar.x_range
    x = torch.randint(xl, xh + 1, (leading, row), generator=g, device=device, dtype=torch.int64)
    return xbar, x


def _bench(xbar, x, device, reps, *, compiled):
    # solve_dc is @torch.compile-decorated in production; force_eager makes it
    # run eagerly so the eager modes are genuinely eager.
    import contextlib

    ctx = contextlib.nullcontext() if compiled else torch.compiler.set_stance("force_eager")
    fn = xbar.core.cim_read
    with ctx:
        fn(x)  # warmup / compile
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        walls = []
        for _ in range(reps):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            fn(x)
            torch.cuda.synchronize(device)
            walls.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return statistics.median(walls), peak


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leading", type=int, default=16384, help="one FFN chunk worth of leading")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    print(f"torch {torch.__version__}  leading={args.leading}  dtype={args.dtype}\n", flush=True)
    rows = []

    # eager-thomas: patch the pcr name back to Thomas, run eager (force_eager).
    nested_solver.solve_block_tridiagonal = solve_block_tridiagonal
    xbar, x = _build(device, dtype, args.leading)
    w, mem = _bench(xbar, x, device, args.reps, compiled=False)
    rows.append(("eager-thomas", w, mem))
    del xbar, x
    torch.cuda.empty_cache()

    # eager-pcr: PCR backend, run eager.
    nested_solver.solve_block_tridiagonal = solve_block_tridiagonal_pcr
    xbar, x = _build(device, dtype, args.leading)
    w, mem = _bench(xbar, x, device, args.reps, compiled=False)
    rows.append(("eager-pcr", w, mem))
    del xbar, x
    torch.cuda.empty_cache()

    # compiled-pcr (scheme A): solve_dc is already @torch.compile-decorated.
    nested_solver.solve_block_tridiagonal = solve_block_tridiagonal_pcr
    torch.compiler.reset()
    xbar, x = _build(device, dtype, args.leading)
    w, mem = _bench(xbar, x, device, args.reps, compiled=True)
    rows.append(("compiled-pcr", w, mem))
    del xbar, x
    torch.cuda.empty_cache()

    # compiled-thomas: Thomas backend, decorated solve_dc compiles (the ~10min
    # cold compile Thomas was rejected for; runtime is the open question).
    nested_solver.solve_block_tridiagonal = solve_block_tridiagonal
    torch.compiler.reset()
    print("[compiled-thomas] cold compile starting (~10 min) ...", flush=True)
    xbar, x = _build(device, dtype, args.leading)
    w, mem = _bench(xbar, x, device, args.reps, compiled=True)
    rows.append(("compiled-thomas", w, mem))

    base = rows[0][1]
    print(f"{'mode':>14} | {'warm_s':>9} | {'peak_gib':>9} | vs eager-thomas", flush=True)
    for name, wall, mem in rows:
        print(f"{name:>14} | {wall:>9.4f} | {mem:>9.3f} | {base / wall:>5.2f}x speed", flush=True)


if __name__ == "__main__":
    main()
