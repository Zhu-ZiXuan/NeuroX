# Compile Contracts — Implementation

The rules every function on the [compiled path](README.md) obeys. They are dynamo-safety invariants, not physics. They hold across every file the macro `matmul` reaches, minus the eager islands. The eager islands ([scheme-a-regional](scheme-a-regional.md)) are the one place these rules are lifted — there you *may* sync, mutate Python state, and run data-dependent loops.

## Invariants

- **No CPU/device sync.** `tensor.item()`, `.tolist()`, `int/float/bool(tensor)`, `.cpu()`, `.numpy()`, `.to("cpu")`, and `print(tensor)` all force a sync dynamo cannot trace. `tensor.shape[i]` / `.size(i)` / `.ndim` return a Python `int` without sync and are safe; `.detach()` (without a following `.item()`) is safe.
- **No Python-state mutation.** No `self.foo = ...` attribute assignment, no mutation of externally-reachable `list` / `dict` / `set`, no `register_buffer` / `register_parameter`, no `module.train()` / `eval()` / `training` toggling inside the region. List accumulation is legal only inside an eager island, never inside a compiled leaf.
- **No tensor-value-dependent control flow.** No `if tensor.max() > 0:` (use `torch.where`), no `while tensor.any():` (use a bounded loop), no `for i in range(int(tensor.sum())):` (vectorize), no `assert <tensor-condition>`. A branch on the **Python type** of an argument or on a Python `bool` is resolved at trace time and is safe — but is exactly the *non-obvious safety* case that warrants a local note.
- **Construct tensors `*_like` an input, not from literals.** Prefer caching at `__init__` / `fabricate` time and reading the buffer; on the path prefer `zeros_like(x)` / `full_like` / `arange`-from-a-buffer over `torch.tensor([...], device=...)`. A cached buffer at the wrong dtype/device still runs — it silently forces a recompile on first divergence, so the symptom is latency, not an exception.
- **`with torch.no_grad()` / `enable_grad()` are safe**, though `enable_grad` makes dynamo more conservative.

## Recompile triggers

> Recompile-key behaviour below is observed on the project's pinned PyTorch and **may change across versions** — re-confirm against the installed version before relying on an edge.

A recompile re-traces a cached graph; it is distinct from a graph break (which drops out of compiled execution for a region). A recompile fires on a change in tensor **shape** (absorbed by `dynamic=True` at the macro entry, pinned by the fixed chunk shape at the regional leaf), **dtype**, **device**, or the concrete **Python type** of an argument. In steady state only the batch shape varies at the macro, and the leaf shape is constant, so a run holds one macro graph per subclass plus the shared leaf graph.

## `fullgraph`

Left at `False`. Library code must tolerate graph breaks: the profiler hooks and the eager island both break by design. `fullgraph=True` is reserved for test code that wants to catch accidental Python sync. `@torch.compiler.disable` is not a license to sync — the eager islands exist for specific, documented reasons; disabling a further function to silence an error hides a real violation. Fix the value-dependent control flow or the host-state mutation instead.
