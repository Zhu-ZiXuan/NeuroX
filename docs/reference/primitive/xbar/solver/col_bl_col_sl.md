# Parallel BL/SL DC solver

## Physical model

Each column contains parallel BL and SL rails, coupled at every row by a single signed cell branch. The columns are electrically independent. A rail is a uniform resistive ladder: row zero connects to its clamp through one segment, and the far end is open. Cell control voltages are prescribed boundaries. Any internal cell nodes are condensed into the terminal branch relation described in [cell](../cell/family.md).

## Governing equations

Consider one column with row index $k=0,\ldots,N_{\mathrm{row}}-1$. Let $a\in\{\mathrm{BL},\mathrm{SL}\}$ identify a rail, with segment resistance $R_{a,\mathrm{seg}}>0$ and conductance $G_{a,\mathrm{seg}}=1/R_{a,\mathrm{seg}}$. The signed current $I_{\mathrm{cell},k}(V_{\mathrm{BL},k},V_{\mathrm{SL},k})$ flows from BL to SL. Its terminal derivatives satisfy

$$
G^{\mathrm{eff}}_{\mathrm{BL},k}
=\frac{\partial I_{\mathrm{cell},k}}{\partial V_{\mathrm{BL},k}}\geq0,
\qquad
G^{\mathrm{eff}}_{\mathrm{SL},k}
=-\frac{\partial I_{\mathrm{cell},k}}{\partial V_{\mathrm{SL},k}}\geq0.
$$

The magnitudes need not be equal: a fixed control voltage can make the branch depend on common-mode voltage as well as the BL-to-SL difference. Taking currents leaving a wire node as positive gives

$$
\begin{aligned}
F_{\mathrm{BL},k}&=I_{\mathrm{cell},k}
+G_{\mathrm{BL},\mathrm{seg}}\left[(V_{\mathrm{BL},k}-V_{\mathrm{BL},k-1})+(V_{\mathrm{BL},k}-V_{\mathrm{BL},k+1})\right],\\
F_{\mathrm{SL},k}&=-I_{\mathrm{cell},k}
+G_{\mathrm{SL},\mathrm{seg}}\left[(V_{\mathrm{SL},k}-V_{\mathrm{SL},k-1})+(V_{\mathrm{SL},k}-V_{\mathrm{SL},k+1})\right].
\end{aligned}
$$

Here $V_{a,-1}=V_{a,\mathrm{port}}$, and the onward-link term is absent at the open far end. The wire equilibrium requires every component of $\mathbf{F}_{\mathrm{node}}$ to vanish.

Each boundary current is positive from its driver into the wire:

$$
I_{a,\mathrm{port}}=G_{a,\mathrm{seg}}(V_{a,\mathrm{port}}-V_{a,0}).
$$

A [clamp driver](../../analog/voltage_driver.md) maps this current to a target voltage $V_{a,\mathrm{target}}=\operatorname{driver}_a(I_{a,\mathrm{port}})$. The boundary residual is

$$
\mathbf{F}_{\mathrm{port}}=
\begin{bmatrix}
V_{\mathrm{BL},\mathrm{target}}-V_{\mathrm{BL},\mathrm{port}}\\
V_{\mathrm{SL},\mathrm{target}}-V_{\mathrm{SL},\mathrm{port}}
\end{bmatrix}.
$$

At wire equilibrium, the BL port current equals the sum of cell currents and the SL port current equals its negative. Away from equilibrium, the boundary-link currents remain the currents implied by the node and port voltages.

## Numerical method

A nested Newton solve eliminates the wire-node unknowns at a fixed port-voltage pair, then corrects the port-voltage pair using the wire response. Each boundary update is followed by another wire solve initialized at the preceding node voltages.

### Initial estimate

The initial port and node voltages equal each rail's nominal reference:

$$
V^{(0)}_{\mathrm{BL},\mathrm{port}}=V_{\mathrm{BL},\mathrm{ref}},
\qquad
V^{(0)}_{\mathrm{SL},\mathrm{port}}=V_{\mathrm{SL},\mathrm{ref}},
$$

$$
V^{(0)}_{\mathrm{BL},k}=V_{\mathrm{BL},\mathrm{ref}},
\qquad
V^{(0)}_{\mathrm{SL},k}=V_{\mathrm{SL},\mathrm{ref}}.
$$

### Coupled wire Newton

Order the unknowns by row, with BL then SL within each row. The wire Jacobian $\mathbf{J}_{\mathrm{node}}$ is block tridiagonal. Its diagonal block at row $k$ is

$$
\mathbf{J}^{\mathrm{diag}}_k=
\begin{bmatrix}
d_{\mathrm{BL},k}+G^{\mathrm{eff}}_{\mathrm{BL},k}&-G^{\mathrm{eff}}_{\mathrm{SL},k}\\
-G^{\mathrm{eff}}_{\mathrm{BL},k}&d_{\mathrm{SL},k}+G^{\mathrm{eff}}_{\mathrm{SL},k}
\end{bmatrix},
\qquad
d_{a,k}=\begin{cases}
2G_{a,\mathrm{seg}},& k<N_{\mathrm{row}}-1,\\
G_{a,\mathrm{seg}},& k=N_{\mathrm{row}}-1.
\end{cases}
$$

The driver link contributes to row zero. A single-row ladder has only that link. Adjacent rows couple through the constant off-diagonal block $\operatorname{diag}(-G_{\mathrm{BL},\mathrm{seg}},-G_{\mathrm{SL},\mathrm{seg}})$. A block Thomas solve computes the correction from

$$
\mathbf{J}_{\mathrm{node}}\,\Delta\mathbf{V}_{\mathrm{node}}=-\mathbf{F}_{\mathrm{node}}.
$$

Each voltage component is clipped to a bounded excursion before being added to the current iterate. This preserves the coupled linearization while limiting movement through nonlinear parts of the branch law.

### Node response to port voltages

The port Jacobian needs the equilibrium response of the port-adjacent nodes to both port voltages:

$$
\mathbf{K}=\frac{\partial\mathbf{V}_{\mathrm{node},0}}{\partial\mathbf{V}_{\mathrm{port}}},
\qquad
\mathbf{V}_{\mathrm{node},0}=\begin{bmatrix}V_{\mathrm{BL},0}\\V_{\mathrm{SL},0}\end{bmatrix},
\qquad
\mathbf{V}_{\mathrm{port}}=\begin{bmatrix}V_{\mathrm{BL},\mathrm{port}}\\V_{\mathrm{SL},\mathrm{port}}\end{bmatrix}.
$$

A change in one port voltage directly perturbs only its rail's row-zero KCL equation. Implicit differentiation gives two systems with the same coupled wire Jacobian:

$$
\mathbf{J}_{\mathrm{node}}\,\mathbf{u}_a=G_{a,\mathrm{seg}}\,\mathbf{e}^{a}_0,
\qquad
\mathbf{K}_{:,a}=\begin{bmatrix}u_{a,\mathrm{BL},0}\\u_{a,\mathrm{SL},0}\end{bmatrix}.
$$

The basis vector $\mathbf{e}^{a}_0$ selects rail $a$ at row zero. Each solve allows every wire node on both rails to readjust; retaining the two row-zero responses gives one column of $\mathbf{K}$. This evaluates the needed sensitivity without forming a dense inverse.

### Port Newton

Let $r_a=\partial V_{a,\mathrm{target}}/\partial I_{a,\mathrm{port}}$ be the signed driver slope. A Thevenin driver has $r_a=-R_{a,\mathrm{out}}\leq0$, including zero for an ideal voltage clamp. With matrix indices zero and one denoting BL and SL, respectively,

$$
\mathbf{J}_{\mathrm{port}}=
\begin{bmatrix}
r_{\mathrm{BL}}G_{\mathrm{BL},\mathrm{seg}}(1-K_{00})-1&-r_{\mathrm{BL}}G_{\mathrm{BL},\mathrm{seg}}K_{01}\\
-r_{\mathrm{SL}}G_{\mathrm{SL},\mathrm{seg}}K_{10}&r_{\mathrm{SL}}G_{\mathrm{SL},\mathrm{seg}}(1-K_{11})-1
\end{bmatrix}.
$$

The direct port-voltage variation contributes the identity in the port-current derivative; node motion subtracts $\mathbf{K}$. Subtracting the boundary voltage from the target contributes the final diagonal $-1$. Solve

$$
\mathbf{J}_{\mathrm{port}}\,\Delta\mathbf{V}_{\mathrm{port}}=-\mathbf{F}_{\mathrm{port}}
$$

and bound each component before updating the port voltages.

### Residual stopping criteria

A column advances as one coupled system, but each row has its own current threshold. Let $\mathcal{N}_{a,k}$ contain node $k$ and the voltages connected to it by a wire segment on rail $a$. This includes the port boundary only at row zero and excludes an onward neighbor at the open far end. Define the local voltage scale, wire-rounding allowance, and current threshold by

$$
V_{a,k,\mathrm{scale}}=\max_{V\in\mathcal{N}_{a,k}}|V|,
\qquad
\eta_k=2\varepsilon\max_a\left(G_{a,\mathrm{seg}}V_{a,k,\mathrm{scale}}\right),
\qquad
\tau_{\mathrm{node},k}=\tau_{I,\mathrm{abs}}+\rho|I_{\mathrm{cell},k}|+\eta_k.
$$

The node solve stops for a column when $\max_a|F_{a,k}|\leq\tau_{\mathrm{node},k}$ at every row $k$. Until then, all rows in the column update together. A large cell current elsewhere in the column does not loosen a node's relative tolerance. The wire allowance scales with local absolute voltages because subtracting nearby node voltages can leave a small drop whose rounding error is amplified by the segment conductance. For an interior KCL row, the sum of absolute voltage coefficients is four times the segment conductance; a voltage-rounding allowance of half the machine epsilon motivates the factor $2\varepsilon$. The same factor conservatively covers the one-link open end, including a single-node rail.

For the port criterion, define scales using the boundary, its first node, and the driver target:

$$
V_{a,\mathrm{portscale}}=\max(|V_{a,\mathrm{port}}|,|V_{a,0}|),
\qquad
V_{\mathrm{port,scale}}=\max_a\left(V_{a,\mathrm{portscale}},|V_{a,\mathrm{target}}|\right),
$$

$$
\tau_{\mathrm{port}}=\tau_{V,\mathrm{abs}}+\rho V_{\mathrm{port,scale}}
+2\varepsilon\max_a\left(|r_a|G_{a,\mathrm{seg}}V_{a,\mathrm{portscale}}\right).
$$

The port solve stops when $\|\mathbf{F}_{\mathrm{port}}\|_\infty\leq\tau_{\mathrm{port}}$. The final term propagates boundary-current rounding through the driver's voltage response. These allowances address wire and port signal conversion; they are not bounds on every source of numerical error in a nonlinear cell or driver.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| $R_{\mathrm{BL},\mathrm{seg}}$, $R_{\mathrm{SL},\mathrm{seg}}$ | Uniform rail link resistance, including the boundary link | MOhm | $>0$ | Extracted |

## Symbols

Shared notation follows [notation conventions](../../../../conventions/notation_conventions.md). Bold quantities collect the scalar components defined below; $\Delta$ denotes a Newton correction, the superscript $(0)$ an initial estimate, and $\|\cdot\|_\infty$ the maximum absolute component.

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $a$ | Rail index, BL or SL | — | — |
| $k$, $j$ | Row indices | — | — |
| $N_{\mathrm{row}}$ | Number of rows | — | Final node-grid extent |
| $V_{\mathrm{BL},k}$, $V_{\mathrm{SL},k}$ | Wire node voltages | V | `v_bl_node__V`, `v_sl_node__V` |
| $V_{\mathrm{BL},\mathrm{port}}$, $V_{\mathrm{SL},\mathrm{port}}$ | Port voltages | V | `v_bl_port__V`, `v_sl_port__V` |
| $\mathbf{V}_{\mathrm{node}}$, $\mathbf{V}_{\mathrm{node},0}$ | All wire voltages, or the port-adjacent pair | V | — |
| $\mathbf{V}_{\mathrm{port}}$ | BL/SL boundary pair | V | — |
| $V_{a,\mathrm{target}}$ | Driver target at the boundary current | V | `ClampDcop.v_port__V` |
| $I_{\mathrm{cell},k}$ | Condensed branch current, positive BL to SL | uA | `ResistiveCellDcop.i__uA` |
| $G^{\mathrm{eff}}_{\mathrm{BL},k}$, $G^{\mathrm{eff}}_{\mathrm{SL},k}$ | Non-negative terminal-derivative magnitudes | uS | `di_dvbl__uS`, negated `di_dvsl__uS` |
| $I_{\mathrm{BL},\mathrm{port}}$, $I_{\mathrm{SL},\mathrm{port}}$ | Current from each driver into its rail | uA | `i_bl_port__uA`, `i_sl_port__uA` |
| $R_{a,\mathrm{seg}}$, $G_{a,\mathrm{seg}}$ | Rail segment resistance and reciprocal conductance | MOhm, uS | `bl_segment_r__MOhm`, `sl_segment_r__MOhm` and their reciprocals |
| $F_{a,k}$, $\mathbf{F}_{\mathrm{node}}$ | Wire-node KCL residuals | uA | `f_bl__uA`, `f_sl__uA` |
| $\mathbf{F}_{\mathrm{port}}$ | BL and SL target-minus-boundary residuals | V | `f_bl_port__V`, `f_sl_port__V` |
| $d_{a,k}$ | Sum of attached wire conductances | uS | — |
| $\mathbf{J}_{\mathrm{node}}$, $\mathbf{J}^{\mathrm{diag}}_k$ | Wire Jacobian and its row-diagonal block | uS | `diag_blocks__uS` and constant off-block |
| $\mathbf{e}^{a}_0$ | Unit basis vector at rail $a$, row zero | — | — |
| $\mathbf{u}_a$ | All-node response to port $a$ | — | Conductance-scaled basis-solve result |
| $\mathbf{K}$ | Port-to-first-node voltage sensitivity | — | `k` |
| $r_a$, $R_{a,\mathrm{out}}$ | Signed driver slope and Thevenin output resistance | MOhm | `ClampDcop.dvport_di__MOhm`; resistance is driver-owned |
| $\mathbf{J}_{\mathrm{port}}$ | Boundary residual Jacobian | — | `port_jacobian` |
| $\mathcal{N}_{a,k}$, $V_{a,k,\mathrm{scale}}$ | Node and attached-neighbor voltage set, and its absolute scale | V | — |
| $\eta_k$ | Local wire-rounding current allowance | uA | `roundoff__uA` |
| $V_{a,\mathrm{portscale}}$ | Absolute voltage scale at a boundary link | V | `v_bl_port_scale__V`, `v_sl_port_scale__V` |
| $V_{\mathrm{port,scale}}$ | Boundary-link and driver-target voltage scale | V | `v_port_scale__V` |
| $\rho$, $\varepsilon$ | Relative tolerance and machine epsilon | — | `rtol`, `torch.finfo(dtype).eps` |
| $\tau_{I,\mathrm{abs}}$, $\tau_{V,\mathrm{abs}}$ | Absolute current and voltage tolerances | uA, V | `node_atol__uA`, `port_atol__V` |
| $\tau_{\mathrm{node},k}$, $\tau_{\mathrm{port}}$ | Per-row current and per-column voltage thresholds | uA, V | `threshold__uA`, `threshold__V` |

## Assumptions, scope & validity

The formulation applies to uniform parallel rails with positive segment resistances, condensed cell branches with the stated derivative signs, and differentiable clamp transfers. An orthogonal two-dimensional wire mesh or a solved control-line network requires a different system.

The implicit sensitivity requires a settled node state and a nonsingular wire Jacobian. Local quadratic Newton convergence requires sufficiently smooth residuals, a nonsingular Jacobian at the root, an initial estimate in its convergence neighborhood, and inactive step bounds. Finite node tolerances limit the accuracy of the port linearization. Bounded steps alone do not establish global convergence or uniqueness for arbitrary nonlinear cell and driver laws.

## Validation

Linear branches with ideal or Thevenin clamps admit an independent dense nodal solution; a single-row column also admits a closed-form circuit solution. Reconstructing wire KCL and boundary residuals from the resulting voltages checks both the electrical state and the stopping criterion. The [adaptive solve validation procedure](../../../../validation/structured_while_solve.md) describes these checks and their operating-envelope coverage.

## References

TODO: cite the Newton and block-tridiagonal solution methods.
