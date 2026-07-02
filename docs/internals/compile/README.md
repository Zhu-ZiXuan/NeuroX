# Compilation

How NeuroX runs under `torch.compile`: where compilation applies, the rules every compile-friendly function obeys, and the scheme that solves the one hard compile problem in the library. Compilation is a software policy with no Reference spec.

NeuroX is written to be compile-*friendly*: primitive methods, including the macro `matmul` (`neurox/macro/xbar/*`), are left eager and undecorated so a caller may `torch.compile` an entire model without the library forcing a policy. The library self-compiles exactly **one** region — the DC-solver leaf `solve_dc` (`@torch.compile(dynamic=False)`) — because that one heavy kernel must compile regardless of caller policy and is the only region shape-stable enough to compile cleanly. See [scheme-a-regional](scheme-a-regional.md).

## Compile-friendly by default

Every primitive method is written to be traceable, so a caller who wraps a model in `torch.compile` gets a working graph, and the library's own regional leaf compiles cleanly. Two facts define this, stated once here rather than per function:

- Library methods obey the dynamo-safety [contracts](contracts.md) (no sync, no host-state mutation, no tensor-value control flow, and so on) by default.
- A handful of **boundaries** deviate from that default — the eager island, the regional leaf, the disabled hooks. See the authoritative boundary map in [scheme-a-regional](scheme-a-regional.md).

A function therefore needs **no** per-module compile note for being ordinary compile-friendly code — that is the default. Module docs call out only the two things that are *not* derivable from the default:

- **Boundaries** — a function that deviates (an eager island, the regional leaf, a disabled hook). The module doc carries a one-line marker pointing here.
- **Non-obvious safety** — a compile-friendly function whose dynamo-safety is not visually obvious (a branch that looks value-dependent but resolves at trace time, a construction that looks like a host literal but is cached). That reasoning is function-specific and stays beside the code, linking to [contracts](contracts.md).

## Schemes

The chunked 1T1R DC solve path — the 1T1R core's chunked read reaching the topology-agnostic `solve_dc` leaf — is the one place the library must self-compile and the one place a naive boundary fails (see [scheme-a-regional](scheme-a-regional.md) §The problem it solves). Three schemes address it, in increasing power and engineering cost. The current scheme is the minimal one that works; the others are recorded as deferred designs to migrate to under named conditions.

| Scheme | Status | Summary | Adopt when |
| --- | --- | --- | --- |
| [A — regional](scheme-a-regional.md) | **Active** | eager forward + eager chunk loop + regionally-compiled `solve_dc` leaf | now (default) |
| [B — de-objectified solver](scheme-b-deobjectified.md) | Deferred | solve hot path as a module-level free function with explicit tensor / scalar args | cross-instance reuse proves fragile, or a predictable cache key is needed, or as a prerequisite for C |
| [C — xbar-read custom op](scheme-c-custom-op.md) | Deferred | wrap the xbar read as a `torch.library.custom_op` opaque node | `fullgraph=True` becomes a hard requirement |

## See also

- [contracts](contracts.md) — the rules every compile-friendly function obeys
- Reference: N/A — compilation is a software policy, no physical spec
