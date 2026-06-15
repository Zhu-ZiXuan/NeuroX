# Compile Policy — Implementation

## Summary

The project-wide rule for `@torch.compile`: it is applied at exactly one layer — the macro entry method `matmul`, on every concrete `XbarMacro` subclass (`neurox/macro/xbar/{ideal,direct,inter_array_slice,intra_array_slice}.py`). Nothing below the macro self-decorates, and two leaf methods opt out with `@torch.compiler.disable`. This document records why the boundary sits there, what the compiled region forbids, and where the intentional graph breaks are. There is no Reference spec — compilation is a software policy, not a physical model.

## Design decisions

- **One compile boundary, at the macro `matmul`.** The macro is the natural unit: by the time `matmul` is called the tile geometry is fixed, the only dynamic shapes are the user batch dimensions, and a single compiled region spans the whole VMM-plus-digital-aggregation pipeline. Decorating below the macro (xbar `vec_mat_mul`, ADC `convert`, the solver, digital `operate`) would only fragment fusion; decorating above it is the caller's choice — the operator-layer `forward` is left undecorated so user code may wrap the whole model with `torch.compile` for outer fusion without nesting against a library-owned region. Rejected: a global module-tree sweep that decorates every leaf (fragments fusion, multiplies recompiles).
- **`dynamic=True`, always.** User batch dimensions vary call-to-call; static-shape mode would recompile on every batch-size change. This is the one decoration parameter the library pins.
- **`fullgraph` left at `False`.** Library code must tolerate graph breaks because the profiler hooks break the graph (see Contracts). `fullgraph=True` is reserved for test code that wants to catch accidental Python sync, not a library default.
- **`mode` and `backend` left at default** (`inductor`). `reduce-overhead` and `max-autotune` are deployment-time tuning, not a library concern.
- **Profiler hooks are disabled rather than made traceable.** `ProfileMixin._log_dynamic_energy` and `_log_latency` (`neurox/common/mixin/profile.py`) carry `@torch.compiler.disable`. They read a `threading.local` side-channel and mutate a Python list — both untraceable by dynamo. The breaks are placed at the end of each primary method, after all kernel math, so fusion inside the kernel is unaffected. Rejected: a tensor-return profiling path that would remove the break — deferred, not adopted, because it complicates the event side-channel.
- **`Offset1T1RXbar.vec_mat_mul` is disabled as a temporary boundary.** `neurox/xbar/_1t1r/offset.py` puts `@torch.compiler.disable` at the top of `vec_mat_mul`. Compiling the inner block produced a >10 min first-call compile dominated by inductor scheduling of the SAR ADC bit-loop (`McsSarAdc.convert`, `neurox/analog/adc/mcs_sar.py`). The disable isolates the Newton-solve-plus-readout chain to run eagerly while the macro-level compile still fuses everything above it. Remove it once the SAR bit-loop is rewritten into a graph-friendly form.

## Contracts & invariants

The **compiled path** is everything reachable from a `@torch.compile`-decorated method, minus the two `@torch.compiler.disable` islands. The following hold across every file on that path.

- **No CPU/device sync.** `tensor.item()`, `.tolist()`, `int/float/bool(tensor)`, `.cpu()`, `.numpy()`, `.to("cpu")`, and `print(tensor)` all force a sync dynamo cannot trace. `tensor.shape[i]` / `.size(i)` / `.ndim` return a Python `int` without sync and are safe; `.detach()` (without a following `.item()`) is safe.
- **No Python-state mutation.** No `self.foo = ...` attribute assignment, no mutation of externally-reachable `list` / `dict` / `set`, no `register_buffer` / `register_parameter`, no `module.train()` / `eval()` / `training` toggling inside the region — dynamo forbids host-state mutation in the compiled region. The solver helpers therefore accumulate into fresh lists rather than writing in-place (`neurox/xbar/solver.py`).
- **No tensor-value-dependent control flow.** No `if tensor.max() > 0:` (use `torch.where`), no `while tensor.any():` (use a bounded loop), no `for i in range(int(tensor.sum())):` (vectorize), no `assert <tensor-condition>`.
- **Construct tensors `*_like` an input, not from literals.** Prefer caching at `__init__` / `fabricate` time and reading the buffer; on the path prefer `zeros_like(x)` / `full_like` / `arange`-from-a-buffer over `torch.tensor([...], device=...)` / `torch.zeros(...)`. The cached buffer's shape/dtype/device must match the runtime path or dynamo recompiles.
- **`with torch.no_grad()` / `enable_grad()` are safe**, though `enable_grad` makes dynamo more conservative; its only use is in `xbar/solver.py:elementwise_diff`.
- **Per-method declaration lives next to the method.** Each primary method's `docs/internals/<path>/<file>.md` carries a one-line `Compile-path: yes (via macro entry).` or `Compile-path: no.`, keeping the constraint beside the code it governs rather than in a single global list that would rot.

## Performance & resources

- **Recompile triggers — distinct from graph breaks.** A graph break drops out of compiled execution for a region; a recompile re-traces the cached graph. The latter is triggered by a change in tensor shape (mitigated by `dynamic=True`), dtype, or device, or a change in the concrete Python type of an argument. In normal use only the batch shape varies, and `dynamic=True` absorbs it, so a steady-state run hits one compiled graph per macro subclass.
- **The disabled SAR boundary is a compile-time, not run-time, trade.** Eager execution of the Newton-solve-plus-readout chain costs nothing in fused-kernel terms that was not already lost to the >10 min inductor scheduling it replaced; the macro-level fusion above `vec_mat_mul` is retained.

## Gotchas

- **`@torch.compiler.disable` is not a license to sync.** The two disabled islands exist for specific, documented reasons (untraceable profiler side-channel; SAR bit-loop compile time). Disabling a third method to "make the error go away" hides a real violation — fix the value-dependent control flow or the host-state mutation instead.
- **`fullgraph=False` masks accidental Python sync.** Because the library default tolerates breaks, an inadvertently introduced `.item()` deep on the path degrades silently into a graph break rather than erroring. Catch it by compiling the suspect region with `fullgraph=True` in a test, not by reading the production decoration.
- **Cached-buffer shape drift recompiles silently.** A buffer fabricated at the wrong dtype/device still runs — it just forces a recompile on first divergence. The symptom is latency, not an exception.

## Known limitations

- **No `fullgraph=True` regression guard in CI.** Whether to expose one as a standing test for accidental Python sync is open. `tests/test_encode.py` checks `torch.compile` equivalence for the encode path only; the macro-entry path is exercised for numerical equivalence by `tests/test_xbar_macro.py` but not under `fullgraph`.
- **The `Offset1T1RXbar.vec_mat_mul` disable is provisional.** It stays until the SAR ADC bit-loop is rewritten into a graph-friendly form; until then the inner solve runs eagerly by design.
- **No tensor-return profiling path.** The single intentional graph break at the profiler hooks remains; a sync-free profiling path that would remove it is deferred.

---

- **Reference**: N/A — compilation is a software policy, no physical spec.
- **Implementation**: `neurox/macro/xbar/ideal.py`, `neurox/macro/xbar/direct.py`, `neurox/macro/xbar/inter_array_slice.py`, `neurox/macro/xbar/intra_array_slice.py`, `neurox/xbar/_1t1r/offset.py`, `neurox/common/mixin/profile.py`, `neurox/analog/adc/mcs_sar.py`
- **Tests**: `tests/test_xbar_macro.py`, `tests/test_encode.py`
- **Decisions**: N/A
