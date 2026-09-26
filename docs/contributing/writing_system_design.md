# Writing system explanations

Explain mechanisms that require understanding more than one component: ownership, physical-state lifetimes, sampling correlation, scheduling, or accounting. A useful page answers a user or extension author's concrete question.

Organize by that question rather than mirroring the code tree. State responsibilities and observable consequences, and explain choices when they affect interpretation or extension. Link an interface only when the reader needs its detailed contract.

Keep single-class rules, signatures, shapes, and failure behavior in the owning docstring. Keep equations and mathematical methods in Reference, task sequences in guides, and contributor rules in conventions. Do not add empty sections, implementation inventories, or speculative future behavior.
