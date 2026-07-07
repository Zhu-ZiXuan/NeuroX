# Scheme C — Xbar-read custom op

**Status: Deferred — not implemented.** The only scheme that makes the xbar read a true opaque node in a `fullgraph=True` parent. Builds on scheme B's functionalization. Adopt only when `fullgraph=True` is a hard requirement — a caller that must `torch.compile(model, fullgraph=True)` with no graph break, a clean export/deploy boundary, or when the eager island's Python-dispatch overhead itself becomes the bottleneck.

## The problem it solves

Scheme A draws an eager island at `solve_array`, so the macro `matmul` graph **breaks** there — fine for `fullgraph=False`, fatal for a parent compiled with `fullgraph=True`. There is no way to keep the chunk loop out of the graph *and* avoid a break except by making the whole xbar read a single opaque node the parent graph neither traces nor breaks on. A custom op is the official mechanism for exactly that.

## The PyTorch mechanism it leans on

> May change across versions — re-confirm before relying on an edge.

- **`torch.library.custom_op`** ([tutorial](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html), [2.x docs](https://docs.pytorch.org/docs/stable/compile/programming_model.custom_ops.html)): wraps a Python function as an operator that `torch.compile` treats as opaque — Dynamo does not trace in **and** Inductor runs it as-is. This is strictly stronger than `torch.compiler.allow_in_graph`, which only blocks the Dynamo front-end while the back-end still traces the body.
- **Fake / meta kernel**: a registered shape-only implementation that returns empty tensors of the correct shape/dtype/device, so the compiler can reason about the op without running it.
- **`torch.library.opcheck`** for registration sanity, and **`torch.library.register_autograd`** if the op must support training (the docs prefer it over mixing `torch.autograd.Function`, which can be silently incorrect under compile).

## Implementation form

Wrap the xbar read as one op; the macro graph then holds a single `neurox::...` node plus its own digital aggregation.

```text
macro.matmul  @torch.compile(dynamic=True, fullgraph possible)
  -> organize_x                                   (compiled-path)
  -> neurox::solve_array_1t1r(state tensors, ...)    opaque to Dynamo and Inductor
  -> readout / digital aggregate                  (compiled-path)
```

Granularity is the main design choice:

- **C1 — wrap `solve_array`.** Hides the solver chunk loop; readout stays on the traceable path; output is the clamp-voltage tensor. Cleanest output shape, but `solve_array`'s profiler emit and energy bookkeeping are side effects that a (functional) op must not hold internally.
- **C2 — wrap `vec_mat_mul`.** Hides core and readout together; output is the ADC-code tensor the macro wants. Larger black box, so the macro can fuse no readout math, and the ADC operating point / rescale enter the op.
- **C3 — wrap only the per-chunk solve body.** Rejected: the chunk loop stays in the parent graph, so the graph holds a custom-op node per chunk and still grows with chunk count — it hides one chunk's internals, not the loop.

Because the op must be functional, it requires scheme B first: all state arrives as tensor arguments, a Python wrapper extracts those tensors from the xbar/core objects, and the op body only computes. Profiler events are emitted **outside** the op (or the op returns energy/latency tensors recorded by an eager wrapper) — Python list mutation must never sit inside a `fullgraph` target.

## Trade-offs

- **vs. schemes A/B.** Gains a real opaque node, so a parent can compile `fullgraph=True`, and gains a clean export/deploy boundary. Loses all fusion inside the op (it is opaque to Inductor) and costs the most engineering: functionalization, fake-kernel registration, profiler relocation, and an explicit autograd policy.
- **C1 vs C2 granularity.** C1 keeps readout fusible in the macro graph and has the simpler output; C2 presents the macro a single xbar-native op at the cost of swallowing readout. Choose by whether readout is worth fusing in the parent.
- **Opaque vs traceable.** Everything inside the op is invisible to the compiler — a deliberate loss of optimization scope in exchange for the black-box boundary.

## Performance and resources (theoretical)

The op runs its body as-is, with no Inductor optimization across its boundary (its interior may still be internally compiled). The payoff is enabling `fullgraph=True` on the parent and a stable opaque boundary, not faster solve execution.

## Risks and failure modes

- **State and side effects.** A custom op is meant to be a functional tensor op. Closing over Python objects yields an unstable graph cache, a fake kernel that cannot express the real state, and possibly a `fullgraph` that appears to succeed while its caching/reuse is unsound. All state must be tensor arguments; side effects must be outside.
- **Autograd.** Without registered autograd, any training/gradient path through the op errors; the physical read must then be inference-only (`no_grad`) or carry an explicitly designed surrogate backward. This must be documented at the op, or users hit a no-autograd error mid-training.
- **Fake-kernel correctness.** A wrong shape/dtype/device in the fake kernel mis-informs the compiler; registration sanity (`opcheck`) does not check the numerical body.
- **Profiler semantics.** Moving energy/latency emission outside the op, or threading it through return tensors, changes how the profiler observes the read — the boundary must be designed so per-VMM accounting stays correct.

## Open questions

- Granularity: C1 (`solve_array`) vs C2 (`vec_mat_mul`).
- Autograd policy: inference-only vs a surrogate backward owned by the operator/algorithm layer.
- Profiler relocation design under a functional op.

## See also

- [scheme A](scheme_a_regional.md), [scheme B](scheme_b_deobjectified.md), [contracts](contracts.md)
