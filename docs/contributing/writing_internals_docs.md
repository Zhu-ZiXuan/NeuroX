# Writing Internals Documents

Internals explains how the code realizes the [Reference](../reference/README.md) specification. Write only what the code cannot tell a reader: why a choice was made, cross-file contracts, performance, gotchas, limitations. Never narrate what the code does — if a sentence is obvious from reading the code, delete it.

## Document template

Use every section, in order. Keep empty headings with `N/A — ...` or `TODO — ...`.

```text
0. Summary            — one sentence on the parts that make up this subsystem; link the Reference spec
1. Design decisions   — each non-obvious choice, its rationale, and the rejected alternatives. The core section
2. Contracts & invariants — call conventions, ownership, shape contracts, torch.compile constraints that hold across files
3. Performance & resources — complexity, memory model, benchmarks, dtype / chunk trade-offs
4. Gotchas            — error-prone behaviour, anti-patterns, "do not treat X as Y"
5. Known limitations  — implementation TODOs, workarounds, and verification coverage gaps (what is deliberately not tested)
```

Do not add a code map (file-by-file class listing): it couples the document to the code and rots on every refactor. Do not add a verification section: put the rationale of a verification strategy in §1 and its coverage gaps in §5; the guarding tests are the footer's Tests entry.

## Never narrate code

Banned: step-by-step restatement of a function's control flow. Example — releasing the DCOP per chunk:

- **Bad:** "Each chunk iteration drops `solver_dcop`; its node tensors are freed before the next chunk allocates."
- **Good:** "Stream by chunk and release the DCOP because the leading batch is ~$10^5$ (im2col × batch × slice × col); materializing all node voltages OOMs. Invariant: per-instance memory stays $O(NB^2)$. Rejected — dense fallback (OOM), PCR materialized shifts (bloat). Chosen — block-Thomas + per-chunk release → peak working set $O(\text{chunk}\times\text{col}\times\text{row})$."

The first restates code; the second gives the reason, the rejected alternatives, and the invariant.

## Footer

End with a horizontal rule and a markdown list — **not** a code block. Link docs with relative `.md` paths; write Implementation and Tests as **inline code, file-level only** — never class / function / line references (those churn). A CI check verifies the paths exist.

```text
---

- **Reference**: [<doc>](<relative .md path>)
- **Implementation**: `neurox/<...>.py`
- **Tests**: `tests/test_<...>.py`
- **Decisions**: [<ADR>](<relative .md path>)   or   TODO — ...
```

## Style

Technical, decision-oriented prose for a contributor. English. Concise. Current-state only.
