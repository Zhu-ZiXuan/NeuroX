"""Diagnose macro-`matmul` recompilation (everything EXCEPT the solver).

The solver leaf (`solve_dc`) sits behind the disabled `cim_read` eager
island, so it is NOT part of the macro `matmul` graph. We force it eager
here (monkeypatch to the undecorated callable) only to skip its ~10 min
cold compile — the macro graph under study is unchanged.

We then drive `InterArraySliceXbarMacro.matmul` (the bert macro type) with
a controlled matrix of (tile geometry, input rank, batch size) and read
`torch._dynamo`'s recompile log + the global unique-graph counter to
attribute each recompile to a specific guard:

  A. geometry sweep, fixed rank/batch  -> isolates buffer-static-shape keying
  B. rank sweep, fixed geometry/batch  -> isolates the ndim guard
  C. batch sweep, fixed geometry/rank  -> checks dynamic=True absorbs size

Run: CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_compile_recompiles
"""

# ruff: noqa: T201

from __future__ import annotations

from pathlib import Path

import torch

from neurox.analog.adc import AdcOperationPoint
from neurox.common import T_ROOM__K
from neurox.macro.xbar import XbarMacro
from neurox.xbar._1t1r.full_jacobian_solver import FullJacobianSolver1T1R
from neurox.xbar._1t1r.nested_solver import NestedSolver1T1R

from example.bert.macro_factory import read_macro_config, read_macro_policy

REPO = Path(__file__).resolve().parent.parent
MACRO_TOML = REPO / "example" / "bert" / "macro.toml"
POLICY_TOML = REPO / "example" / "bert" / "macro.policy.toml"


def _force_solver_eager() -> None:
    """Replace the `@torch.compile`d `solve_dc` with its original callable."""
    for cls in (NestedSolver1T1R, FullJacobianSolver1T1R):
        orig = cls.solve_dc._torchdynamo_orig_callable  # type: ignore[attr-defined]
        cls.solve_dc = orig  # type: ignore[assignment]


def _force_macro_eager_backend() -> None:
    """Re-wrap the macro `matmul` with the `eager` backend.

    Dynamo still traces, guards, and counts recompiles, but Inductor codegen
    (the slow, single-threaded ~minutes-per-graph cost) is skipped — so the
    recompile *structure* (ndim / shape / object guards) is observed fast.
    The original is `@torch.no_grad` wrapped; we drop that and drive the body
    under an explicit `torch.no_grad()` in `_run`.
    """
    from neurox.macro.xbar.inter_array_slice import InterArraySliceXbarMacro

    raw = getattr(InterArraySliceXbarMacro.matmul, "_torchdynamo_orig_callable", None)
    if raw is None:
        # Production state: the macro matmul is eager (not self-compiled), so
        # there is nothing to re-wrap. The probe then measures ~0 macro graphs,
        # which is the regression check that the explosion stays fixed.
        print("[probe] macro matmul is eager (not self-compiled) — expect ~0 macro graphs", flush=True)
        return
    InterArraySliceXbarMacro.matmul = torch.compile(raw, backend="eager", dynamic=True)  # type: ignore[assignment]


def _build_macro(n: int, k: int, device: torch.device) -> XbarMacro:
    macro = XbarMacro.from_config(
        config=read_macro_config(MACRO_TOML),
        policy=read_macro_policy(POLICY_TOML),
        name=f"m_{n}x{k}",
        w_logical_shape=(n, k),
        dtype=torch.float32,
        T__K=T_ROOM__K,
        ideal_xbar=False,
    )
    macro = macro.to(device)
    macro.fabricate()
    macro.program(torch.zeros(n, k, dtype=torch.int32, device=device))
    return macro


def _run(macro: XbarMacro, leading: tuple[int, ...], k: int, device: torch.device) -> None:
    op = AdcOperationPoint(adc_mode=0, adc_bits=macro.adc_max_bits)
    x = torch.zeros(*leading, 1, k, dtype=torch.int32, device=device)  # (..., M=1, K)
    with torch.no_grad():
        macro.matmul(x, adc_operation_point=op)


def _graphs() -> int:
    return int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _force_solver_eager()
    _force_macro_eager_backend()
    # Raise the limits so the probe COUNTS every distinct graph instead of
    # early-stopping into eager at 8 (the very fallback we are diagnosing).
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 64
    torch._logging.set_logs(recompiles=True)

    # Tile = 64x64. Geometries chosen for minimal tiles but distinct (Tc, Tr).
    g11 = _build_macro(64, 64, device)   # Tc=1, Tr=1
    g21 = _build_macro(128, 64, device)  # Tc=2, Tr=1
    g12 = _build_macro(64, 128, device)  # Tc=1, Tr=2

    def mark(tag: str) -> int:
        n = _graphs()
        print(f"\n========== {tag}  (unique_graphs so far={n}) ==========", flush=True)
        return n

    # --- A: geometry sweep, fixed rank-4 input (batch=2, seq=3) ---
    a0 = mark("A: geometry sweep, rank-4 (2,3,1,K)")
    _run(g11, (2, 3), 64, device)
    a1 = _graphs(); print(f"[A] g(1,1) -> +{a1 - a0}", flush=True)
    _run(g21, (2, 3), 64, device)
    a2 = _graphs(); print(f"[A] g(2,1) -> +{a2 - a1}", flush=True)
    _run(g12, (2, 3), 128, device)
    a3 = _graphs(); print(f"[A] g(1,2) -> +{a3 - a2}", flush=True)

    # --- B: rank sweep, fixed geometry g(1,1) ---
    b0 = mark("B: rank sweep on g(1,1)")
    _run(g11, (2,), 64, device)  # rank-3 (2,1,K) -- pooler/classifier shape
    b1 = _graphs(); print(f"[B] rank-3 (2,1,K) -> +{b1 - b0}", flush=True)
    _run(g11, (2, 5), 64, device)  # rank-4 again, new batch dims
    b2 = _graphs(); print(f"[B] rank-4 (2,5,1,K) -> +{b2 - b1}", flush=True)

    # --- C: batch-size sweep, fixed geometry g(1,1), fixed rank-4 ---
    c0 = mark("C: batch-size sweep on g(1,1), rank-4")
    _run(g11, (4, 7), 64, device)
    c1 = _graphs(); print(f"[C] (4,7,1,K) -> +{c1 - c0}", flush=True)
    _run(g11, (8, 9), 64, device)
    c2 = _graphs(); print(f"[C] (8,9,1,K) -> +{c2 - c1}", flush=True)

    print(f"\n==== TOTAL unique_graphs = {_graphs()} ====", flush=True)
    print("counters[stats]:", dict(torch._dynamo.utils.counters["stats"]), flush=True)


if __name__ == "__main__":
    main()
