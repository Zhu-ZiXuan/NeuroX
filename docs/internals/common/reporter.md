# Reporter

`neurox/common/reporter.py` turns one model's modules and one book of energy records into report rows. A record is what an emitter hands the ledger; an entry is one aggregated row a report yields, and the reporter is the only place the first becomes the second. How records are laid out, named, and collected is the [profiler](profiler.md)'s.

## Design decisions

- **A name needs a root, and a ledger has none.** A ledger captures whatever emits while it is open, so its records carry names but nothing to resolve them against. The reporter supplies that root by binding one model at construction: a name becomes a row only where a model explains it, and every figure a measurement comes to is taken here rather than off the book.
- **Binding walks the model once, and that walk is the gate.** The walk collects the static rows and checks that every profile-capable module the model holds carries exactly the name this very walk hands out, so a missing stamp, a stamp from another tree, and one instance bound at two locations are all caught before a single record is read. It keeps duplicate paths — `remove_duplicate=False`, as the stamping walk does — so a second binding site is met under a name no stamp can match rather than being silently skipped. Constructing the reporter before the measurement is therefore the canonical order.
- **Static is settled at binding; dynamic is asked per book.** Area and leakage follow from the tree alone, so `static_entries` and `static` are built by that one walk and never recomputed. Every dynamic view — `dynamic_entries`, `by_name`, `by_group`, `total_dynamic_energy__fJ` — takes the profiler holding the records as an argument, so one reporter serves any number of measurements of the same model and nothing cached can go stale against a second book. Rebuilding a reporter is safe at any time: the per-module properties the walk reads are plain arithmetic over init-fixed data.
- **One book against one model is the caller's discipline.** Nothing ties a ledger to a model — the ledger has no root, and the reporter is handed a finished book — so pairing a measurement with the model that explains it is the caller's. What the reporter can refuse, it does: any single record the bound model cannot name stops the report. A subtree is reported by binding a model that holds its emitters and, where the book is wider than that model, by filtering the records before reporting them.
- **An unnameable record is an error, not a synthesized label.** A record naming a module the bound model does not hold stops the report. Inventing a label for it would put a row in the dump that no model explains, and quietly fold hardware from another tree into a total.
- **A channel joins the path as a virtual submodule.** A record's channel is appended to its emitter's dotted name as one further segment, so a composite's separately billed branches read as rows beneath it without a module instance of their own. That it must be able to be one segment is the whole of its naming rule: a channel carrying a dot is refused, and so is one whose virtual name collides with a real module of the bound model, because a row naming both a real child and a virtual branch would silently merge two pieces of hardware. The bound model's own name is empty, so a branch the model bills itself reads with a leading empty segment — a row name starting with a dot — which is what tells it apart from a top-level child of the same name.
- **Grouping beyond a row name is a caller's dict.** A coarser figure is stated over row names — the very vocabulary `by_name` returns, virtual channel rows included — so a caller folds any set of rows into one label without the reporter guessing what belongs together. The mapping is total over what was measured and need not be total over the model: a measured row the grouping does not cover is an omission the reporter refuses to hide, while a mapped row no record used contributes nothing and is not reported, so one grouping policy outlives one measurement and may cover more rows than a given run exercises.
- **Every view reduces through one path, and pays one sync.** Per-row, per-group and total alike stack the whole book and transfer once, so no two figures over one book can disagree about what it came to, and each call costs exactly one host sync however many records it covers. A record stays on the device it was parked on until a view asks for it.
- **The dump is aligned plain text in fixed units.** `render` writes fJ, um2 and uW verbatim and auto-scales nothing, so two dumps of the same model compare line by line and a diff of them is readable; columns are padded to their widest cell, the name column left-aligned and the numbers right-aligned. It is an inspection surface rather than a data format — a consumer wanting numbers calls the views. With no profiler the dump is the static table alone, which is what a model inspected before it is ever run can offer.

## Contracts & invariants

- **Report after the context.** A view reduces whatever the book holds, and a clean exit is what parks the records on the ledger's device.
- **Each view states its own row order.** `dynamic_entries` is ordered by descending energy; `by_name` and `by_group` are in first-contribution order, following the book; `static_entries` is in the bound model's traversal order. Rows are paired across views by name, never by position.
- **A row name is one string, whatever produced it.** A module row and a virtual channel row are both dotted paths in the same namespace, so the grouping vocabulary, the dump, and the per-name view never have to tell them apart.

## Performance & resources

Binding costs one module-tree walk, $O(\mathrm{circuits})$ derived-property reads and no tensor allocation. A dynamic view costs one stack of the book plus one host transfer — one sync regardless of record count — and $O(\mathrm{records})$ Python work to merge; nothing is memoized, so a caller wanting several figures from one book keeps what it took rather than asking twice.

## Gotchas

- **The bound model's own un-channelled rows carry an empty name.** The model itself is named by the empty string, so a record it emits without a channel resolves to no segments and its row key is `""`. Bill such a branch under a channel to give it a readable row.
- **A rewired model needs a fresh stamping and a fresh reporter.** Names come from a walk, so surgery after binding leaves both the stamps and the static rows describing another shape of tree.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/reporter.py`
- **Tests**: `tests/common/test_reporter.py`
