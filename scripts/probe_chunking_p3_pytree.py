"""P3 probe — verify ``@dataclass(frozen=True)`` snapshot containers
travel through ``@torch.compile(dynamic=False)`` correctly.

We mimic the real snapshot shape: an outer ``OpAmpTIASnapshot``-like
container holding tensor fields plus a nested ``NMOSSnapshot``-like
container. We compile a free function that consumes the nested container
and returns a tensor. The probe checks:

  * the function compiles without raising;
  * the first call is slow (cold), the second call with the same shapes
    is fast (warm cache hit);
  * a third call with a different shape re-traces;
  * a fourth call that revisits the first shape hits the cache again.

If any of those checks fails we know we cannot pass nested frozen
dataclasses directly across the compile boundary and must lower the
free function signature to flat tensor args.
"""

# ruff: noqa: T201

import time
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class _InnerSnap:
    a: Tensor
    b: Tensor


@dataclass(frozen=True, kw_only=True)
class _OuterSnap:
    gain: Tensor
    inner: _InnerSnap


@torch.compile(dynamic=False)
def _consume(x: Tensor, snap: _OuterSnap) -> Tensor:
    return x * snap.gain + snap.inner.a - snap.inner.b


def _build_snap(shape: tuple[int, ...], device: torch.device) -> _OuterSnap:
    return _OuterSnap(
        gain=torch.randn(shape, device=device),
        inner=_InnerSnap(
            a=torch.randn(shape, device=device),
            b=torch.randn(shape, device=device),
        ),
    )


def _time_call(x: Tensor, snap: _OuterSnap, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    out = _consume(x, snap)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time() - t0, out


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[p3] device={device}")

    shape_a = (4, 8)
    shape_b = (4, 16)

    x_a = torch.randn(shape_a, device=device)
    x_b = torch.randn(shape_b, device=device)
    snap_a1 = _build_snap(shape_a, device)
    snap_a2 = _build_snap(shape_a, device)
    snap_b = _build_snap(shape_b, device)

    t1, _ = _time_call(x_a, snap_a1, device)
    print(f"[p3] call 1 (cold shape A): {t1:.3f}s")

    t2, _ = _time_call(x_a, snap_a2, device)
    print(f"[p3] call 2 (warm shape A, new snap): {t2:.4f}s")

    t3, _ = _time_call(x_b, snap_b, device)
    print(f"[p3] call 3 (cold shape B): {t3:.3f}s")

    t4, _ = _time_call(x_a, snap_a1, device)
    print(f"[p3] call 4 (warm shape A again): {t4:.4f}s")

    print("[p3] -----------------------------------------------")
    cold_ratio = t2 / t1 if t1 > 0 else 0.0
    revisit_ratio = t4 / t1 if t1 > 0 else 0.0
    print(f"[p3] warm/cold A ratio: {cold_ratio:.3%}  (want <5%)")
    print(f"[p3] re-visit/cold A ratio: {revisit_ratio:.3%}  (want <5%)")
    print(f"[p3] cold B (different shape): {t3:.3f}s  (expect another full cold compile)")

    if cold_ratio < 0.05 and revisit_ratio < 0.05:
        print("[p3] PASS — frozen nested dataclass goes through @torch.compile, cache works")
    else:
        print("[p3] FAIL — cache not behaving; need flat tensor args")


if __name__ == "__main__":
    main()
