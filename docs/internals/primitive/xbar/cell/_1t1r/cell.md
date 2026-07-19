# 1T1R cell base

`XbarCell1t1r` (`neurox/primitive/xbar/cell/_1t1r.py`) is the abstract intermediate base of the 1T1R cell models; the shared family and container contracts live at [xbar cell base](../README.md). It owns the substrate every 1T1R model shares — the four per-cell node-to-ground capacitances and the grounded-cap energy formula — while the branch physics (`snapshot` / `program` / `solve_branch` / `solve_dc`) stays abstract for the leaves.

## Design decisions

- **One DCOP type for the whole 1T1R family.** Both leaves return the same `XbarCell1t1rDcop` (branch triple plus `v_x__V`), and `XbarCell1t1rResiduals` is shared, so the array's energy path and diagnostics consume one type regardless of which model is configured. Only the snap forks per leaf: the base `XbarCell1t1rSnap` carries `v_wl__V` (every 1T1R model consumes the WL drive), and each leaf's snap subclass adds the per-call state its own solve needs.
- **Shared substrate in the base `__init__`.** The base copies the four node caps `c_{bl,x,sl,wl}__fF` from the config onto the instance, so a leaf never re-derives them. `XbarCell1t1rConfig` carries exactly these four fields and their non-negativity validation; leaf configs extend it with their model-specific fields (state map and device sizing on the Detail config, conductance tables on the Linear config).
- **Shared `dynamic_energy`, grounded node caps only.** The grounded-cap formula lives once in the base and reads only the four base-owned node caps plus the operating-point voltages — no leaf-supplied cap attributes, no coupled terms. Both leaves consume it identically; only `v_x__V` differs by each leaf's condensation.
- **`w_states` is a per-leaf derivation.** The base declares the attribute; each leaf derives and sets it in `__init__` from its own config (Detail: state-map length; Linear: table row count), so the array's `self.w_states = self.cell.w_states` contract is model-agnostic.
- **Policy is an abstract marker.** `XbarCell1t1rPolicy` is an empty ABC; each leaf declares its own policy subclass (a composite of device policies, or empty for a deterministic model). The concrete policy type is the TOML discriminator that pairs with the leaf's config type.
- **Explicit ABC per house convention.** The base declares `ABC` and re-binds the family generics as `XbarCell[XbarCell1t1rSnap, XbarCell1t1rDcop]`; leaves narrow their snap parameter internally.

## Contracts & invariants

- **Leaves set `w_states` in `__init__`.** The array reads `cell.w_states` right after construction; a leaf that fails to set it breaks at array build time.
- **`program` speaks state indices.** Whatever a leaf materializes internally, its `program(w_state_idx)` consumes an index tensor at `inst_shape` in `[0, w_states)`, so the array's programming path is model-agnostic.

---

- **Reference**: [cell](../../../../../reference/primitive/xbar/cell/_1t1r/cell.md)
- **Implementation**: `neurox/primitive/xbar/cell/_1t1r.py`
- **Tests**: exercised through the leaf tests (footer of each leaf page)
