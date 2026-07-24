# 1T1R cell base

`XbarCell1t1r` (`neurox/primitive/xbar/cell/_1t1r.py`) is the abstract intermediate base of the 1T1R cell models; the shared family and container contracts live at [xbar cell base](../base.md). It owns the substrate every 1T1R model shares — the four per-cell node-to-ground capacitances and the grounded-cap energy formula — while the branch physics (`snapshot` / `program` / `solve_branch` / `solve_dc`) stays abstract for the leaves.

## Design decisions

- **One DCOP type for the whole 1T1R family.** Both leaves return the same `XbarCell1t1rDcop` (branch triple plus `v_x__V`), and `XbarCell1t1rResiduals` is shared, so the array's energy path and diagnostics consume one type regardless of which model is configured. Only the snap forks per leaf: the base `XbarCell1t1rSnap` carries `v_wl__V` (every 1T1R model consumes the WL drive), and each leaf's snap subclass adds the per-call state its own solve needs.
- **Shared substrate configuration.** `XbarCell1t1rConfig` carries the four node caps `c_{bl,x,sl,wl}__fF` and their non-negativity validation; leaf configs extend it with their model-specific fields (state map and device sizing on the Detail config, conductance tables on the Linear config). The base does not duplicate these immutable scalar values on the module.
- **Shared `compute_dynamic_energy`, grounded node caps only.** The grounded-cap formula lives once in the base and reads the four node caps directly from `self.config` plus the operating-point voltages — no leaf-supplied cap attributes, no coupled terms. Both leaves consume it identically; only `v_x__V` differs by each leaf's condensation. The cell owns and bills the switching energy of all four cell nodes — the BL, the internal X, the SL, and the WL NMOS gate it owns (`e_wl__fJ = c_wl · v_wl²`); the WL *wire* charge is billed by the owning array (cell gate / array wire).
- **`w_state_num` is a per-leaf derivation.** The base declares the attribute; each leaf derives and sets it in `__init__` from its own config (Detail: state-map length; Linear: table row count), and the array's `w_state_num` property forwards the value without knowing the concrete cell model.
- **Policy is an abstract marker.** `XbarCell1t1rPolicy` is an empty ABC; each leaf declares its own policy subclass (a composite of device policies, or empty for a deterministic model). The concrete policy type is the TOML discriminator that pairs with the leaf's config type.
- **Explicit ABC per house convention.** The base declares `ABC` and keeps its snap type generic under the `XbarCell1t1rSnap` bound; each leaf binds that parameter to its concrete snap type while the whole family shares `XbarCell1t1rDcop`.
- **The 1T1R family root carries the registry and `from_config`.** `XbarCell1t1r` owns the `from_config(*, config, policy, inst_shape, dtype, T__K)` factory; each leaf registers its `(LeafConfig, LeafPolicy)` pair. The owning array therefore resolves the cell and rejects a mismatched pair before construction, with no leaf-level `isinstance` narrowing. The universal `XbarCell` base stays a pure solver-facing contract.

## Contracts & invariants

- **Leaves set `w_state_num` in `__init__`.** The array reads `cell.w_state_num` right after construction; a leaf that fails to set it breaks at array build time.
- **`program` speaks state indices.** Whatever a leaf materializes internally, its `program(w_state_idx)` consumes an index tensor at `inst_shape` in `[0, w_state_num)`, so the array's programming path is model-agnostic.

---

- **Reference**: [cell](../../../../../reference/primitive/xbar/cell/_1t1r/cell.md)
- **Implementation**: `neurox/primitive/xbar/cell/_1t1r.py`
- **Tests**: exercised through the leaf tests (footer of each leaf page)
