# Compile contracts

The rules every compile-friendly function obeys. They are dynamo-safety invariants, not physics. They hold across the regionally-compiled `solve_dc` leaf and across every file the macro `matmul` reaches — the forward the library keeps traceable so a caller may `torch.compile` it — minus the eager islands. The eager islands (scheme_a_regional) are the one place these rules are lifted — there you *may* sync, mutate Python state, and run data-dependent loops.

## Compile-friendly by default

Every primitive method is written to be traceable, so a caller who wraps a model in `torch.compile` gets a working graph, and the library's own regional leaf compiles cleanly. Two facts define this, stated once here rather than per function:

- Library methods obey the dynamo-safety invariants below (no sync, no host-state mutation, no tensor-value control flow, and so on) by default.
- A handful of **boundaries** deviate from that default — the eager island, the regional leaf, the disabled hooks. See the authoritative boundary map in scheme_a_regional.

A function therefore needs **no** per-module compile note for being ordinary compile-friendly code — that is the default. Module docs call out only the two things that are *not* derivable from the default:

- **Boundaries** — a function that deviates carries a one-line marker pointing to scheme_a_regional.
- **Non-obvious safety** — a compile-friendly function whose dynamo-safety is not visually obvious (a branch that looks value-dependent but resolves at trace time, a construction that looks like a host literal but is cached). That reasoning is function-specific and stays beside the code, linking here.

## Invariants

- **No CPU/device sync.** `tensor.item()`, `.tolist()`, `int/float/bool(tensor)`, `.cpu()`, `.numpy()`, `.to("cpu")`, and `print(tensor)` all force a sync dynamo cannot trace. `tensor.shape[i]` / `.size(i)` / `.ndim` return a Python `int` without sync and are safe; `.detach()` (without a following `.item()`) is safe.
- **No Python-state mutation.** No `self.foo = ...` attribute assignment, no mutation of externally-reachable `list` / `dict` / `set`, no `register_buffer` / `register_parameter`, no `module.train()` / `eval()` / `training` toggling inside the region. List accumulation is legal only inside an eager island, never inside a compiled leaf.
- **No tensor-value-dependent control flow.** No `if tensor.max() > 0:` (use `torch.where`), no `while tensor.any():` (use a bounded loop), no `for i in range(int(tensor.sum())):` (vectorize), no `assert <tensor-condition>`. A branch on the **Python type** of an argument or on a Python `bool` is resolved at trace time and is safe — but is exactly the *non-obvious safety* case that warrants a local note.
- **Construct tensors `*_like` an input, not from literals.** Prefer caching at `__init__` / `fabricate` time and reading the buffer; on the path prefer `zeros_like(x)` / `full_like` / `arange`-from-a-buffer over `torch.tensor([...], device=...)`. A cached buffer at the wrong dtype/device still runs — it silently forces a recompile on first divergence, so the symptom is latency, not an exception.
- **`with torch.no_grad()` / `enable_grad()` are safe**, though `enable_grad` makes dynamo more conservative.

## Recompile triggers

> Recompile-key behaviour below is observed on the project's pinned PyTorch and **may change across versions** — re-confirm against the installed version before relying on an edge.

A recompile re-traces a cached graph; it is distinct from a graph break (which drops out of compiled execution for a region). A recompile fires on a change in tensor **shape** (pinned to the fixed chunk shape at the regional leaf; absorbed by `dynamic=...` if a caller compiles the forward), **dtype**, **device**, or the concrete **Python type** of an argument. In steady state the leaf shape is constant, so a run holds just the shared leaf graph — plus whatever a caller's own model-level compile adds.

## `fullgraph`

Left at `False`. Library code must tolerate graph breaks: the profiler hooks and the eager island both break by design. `fullgraph=True` is reserved for test code that wants to catch accidental Python sync. `@torch.compiler.disable` is not a license to sync — the eager islands exist for specific, documented reasons; disabling a further function to silence an error hides a real violation. Fix the value-dependent control flow or the host-state mutation instead.
