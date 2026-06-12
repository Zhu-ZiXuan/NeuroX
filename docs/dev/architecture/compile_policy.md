# `@torch.compile` Policy

This document records the project-wide rules for using `@torch.compile`, the forbidden behaviours on the compiled path, and the intentional graph breaks.

## Where the compile boundary lives

The project applies `@torch.compile` at exactly one layer: the **macro entry method**.

Concretely:

- Every concrete `XbarMacro` subclass's `matmul` method is decorated `@torch.no_grad()` / `@torch.compile(dynamic=True)`.
- **Nothing below the macro is decorated.** Xbar `vec_mat_mul`, ReadOut `readout`, ADC `convert`, SwitchCap `sample_and_accumulate`, AnalogMux `transport`, OpAmpTIA `solve_dc`, `Solver1T1R.solve_dc`, digital `operate`, etc. are plain methods.

The macro layer is the natural unit boundary: tile geometry is known by the time `matmul` is called, dynamic shapes only enter through user batch dimensions, and one compiled region contains the whole VMM + digital aggregation pipeline.

Below the macro, decorating individual functions would only fragment fusion. Above the macro, the caller decides what to compile (the operator layer's `forward` is undecorated; PyTorch user code can wrap the whole model with `torch.compile` if it wants outer fusion).

## Decoration parameters

When a method is allowed to self-decorate (currently only `XbarMacro.matmul` on every concrete subclass):

- `dynamic=True` — always. User batch dimensions vary call-to-call; static-shape mode would recompile on every batch-size change.
- `fullgraph` — default `False`. Library code must allow graph breaks because `_log_dynamic` produces one (see below). Set `fullgraph=True` only in test code that wants to catch accidental Python sync.
- `mode` — leave default. `reduce-overhead` and `max-autotune` are deployment-time tuning, not library defaults.
- `backend` — leave default (`inductor`).

## Forbidden behaviours on the compiled path

The "compiled path" is everything reachable from a `@torch.compile`-decorated method. Inside this region, the following are forbidden because dynamo cannot trace them and they break fusion:

### CPU / device synchronisation

- `tensor.item()`, `tensor.tolist()`, `int(tensor)`, `float(tensor)`, `bool(tensor)` — force CPU sync.
- `tensor.cpu()`, `tensor.numpy()`, `tensor.to("cpu")` — force device migration.
- `print(tensor)` — implicit CPU sync.

### Python-state mutation

- Assigning to module attributes (`self.foo = ...`) — dynamo does not allow host-state mutation inside the compiled region.
- Mutating Python `list` / `dict` / `set` instances reachable from outside the call.
- `register_buffer(...)` / `register_parameter(...)` — changes the module tree.
- Toggling `module.training` / calling `module.eval()` / `module.train()`.

### Tensor-value-dependent control flow

- `if tensor.max() > 0:` — use `torch.where`.
- `while tensor.any():` — replace with a bounded loop.
- `for i in range(int(tensor.sum())):` — vectorise.
- `assert <tensor-condition>` — likewise.

### Tensor construction on the path (avoid)

- `torch.tensor([1, 2, 3], device=...)` — prefer caching at `__init__` / `fabricate` time and reading the buffer.
- `torch.zeros(...)`, `torch.full(...)`, `torch.arange(...)` — prefer `*_like(existing)` so the device / dtype follow the input.

The shape / dtype / device of the cached buffer must match the runtime path; otherwise dynamo will trigger a recompile.

## Allowed exceptions

- `@torch.compiler.disable` on `ProfileMixin._log_dynamic`. The break is local — it happens at the end of each primary method, after all the kernel math, so fusion inside the kernel is unaffected. Necessary because the profiler reads `threading.local` and mutates a Python list; both are untraceable by dynamo. See [`profiler_and_ppa.md`](profiler_and_ppa.md) for the trade-off discussion.
- `@torch.compiler.disable` on `Offset1T1RXbar.vec_mat_mul` (`neurox/xbar/_1t1r/offset.py`). This is a **temporary intentional boundary**: an earlier attempt to compile the inner block produced a > 10 min first-call compile dominated by inductor scheduling of the SAR ADC's bit-loop (`McsSarAdc.convert`). The disable is at the top of `vec_mat_mul` so the macro-level compile still fuses everything *above* it; the Newton solve + readout chain runs eagerly, which is the project default until the SAR bit-loop is rewritten to a graph-friendly form. Remove this `@disable` when the SAR fix lands.
- `tensor.shape[i]` / `tensor.size(i)` / `tensor.ndim` return Python `int` without CPU sync — safe.
- `.detach()` (without `.item()`) — safe.
- `with torch.no_grad():` / `with torch.enable_grad():` blocks — safe, although `enable_grad` makes dynamo more conservative; the only current use is inside `xbar/solver.py:elementwise_diff`.

## Recompile triggers (avoid when possible)

The following do not break compilation but trigger a recompile of the cached graph:

- Tensor shape changes (mitigated by `dynamic=True`).
- Tensor dtype changes.
- Tensor device changes.
- Changes in concrete Python types passed as arguments.

The macro entry uses `dynamic=True` so user batch dimensions do not recompile. Other parameters (dtype, device) should not change between calls of the same compiled function in normal use.

## Per-method compile-path declaration

Each primary method's `docs/dev/modules/<path>/<file>.md` declares whether it is on the compiled path. The declaration is one line under the method description:

```
Compile-path: yes (via macro entry).
```

or

```
Compile-path: no.
```

This puts the runtime-path constraint next to the method it applies to, instead of maintaining a separate global list.

## Future work (deferred)

- Whether to expose `fullgraph=True` as a regression test for accidental Python sync.
- Whether to provide a tensor-return profiling path that removes the only graph break.

Both are open and not in scope today.
