# Compilation — Implementation

How NeuroX runs under `torch.compile`: where compilation applies, the rules every compiled-path function obeys, and the scheme that solves the one hard compile problem in the library. Compilation is a software policy with no Reference spec.

NeuroX is written to run under `torch.compile`. The library does not wrap whole models itself — the operator-layer `forward` is left undecorated so a caller may `torch.compile` an entire model. The one library-owned, user-visible compile entry is the macro method `matmul` (`neurox/macro/xbar/*`), decorated `@torch.compile(dynamic=True)`.

## The compiled path

The **compiled path** is everything reachable from `macro.matmul`, minus a few deliberate **eager islands**. Two facts define it, and both are stated once here rather than per function:

- Everything below `matmul` is on the compiled path **by default**.
- Every compiled-path function obeys the dynamo-safety [contracts](contracts.md) (no sync, no host-state mutation, no tensor-value control flow, and so on).

A function therefore needs **no** per-module compile note for being an ordinary compiled-path function — that is the default and is fully implied by the two facts above. Module docs call out only the two things that are *not* derivable from the default:

- **Boundaries** — a function that deviates from the default (an eager island, a separately-compiled regional leaf, a disabled hook). The authoritative list lives in [scheme-a-regional](scheme-a-regional.md); the module doc carries a one-line marker pointing here.
- **Non-obvious safety** — a function on the compiled path whose dynamo-safety is not visually obvious (a branch that looks value-dependent but resolves at trace time, a construction that looks like a host literal but is cached). That reasoning is function-specific and stays beside the code, linking to [contracts](contracts.md).

## Boundaries at a glance

| Layer | Decoration | Role |
| --- | --- | --- |
| macro `matmul` | `@torch.compile(dynamic=True)` | user-visible entry; only batch shapes vary |
| xbar `vec_mat_mul` | on the compiled path | index / readout math, traced by `matmul` |
| core `cim_read` | `@torch.compiler.disable(recursive=False)` | eager island that owns the chunk loop |
| solver `solve_dc` | `@torch.compile(dynamic=False)` | regional leaf; fixed chunk shape, compiled once, reused |
| profiler hooks | `@torch.compiler.disable` | side-channel writes; intentional graph break |

## Schemes

The chunked 1T1R DC solver is the one place a naive compile boundary fails (see [scheme-a-regional](scheme-a-regional.md) §problem). Three schemes address it, in increasing power and engineering cost. The current scheme is the minimal one that works; the others are recorded as deferred designs to migrate to under named conditions.

| Scheme | Status | Summary | Adopt when |
| --- | --- | --- | --- |
| [A — regional](scheme-a-regional.md) | **Active** | eager chunk loop + regionally-compiled `solve_dc` leaf | now (default) |
| [B — de-objectified solver](scheme-b-deobjectified.md) | Deferred | solve hot path as a module-level free function with explicit tensor / scalar args | cross-instance reuse proves fragile, or a predictable cache key is needed, or as a prerequisite for C |
| [C — xbar-read custom op](scheme-c-custom-op.md) | Deferred | wrap the xbar read as a `torch.library.custom_op` opaque node | `fullgraph=True` becomes a hard requirement |

## See also

- [contracts](contracts.md) — the rules every compiled-path function obeys
- Reference: N/A — compilation is a software policy, no physical spec
