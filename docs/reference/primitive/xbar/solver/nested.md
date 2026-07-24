# Parallel BL/SL DC solver

The DC operating point of a crossbar array with **parallel BL/SL rails** is found by damped Newton iteration over the two wire ladders and the two clamp boundaries. Every array site condenses to a single signed two-terminal branch and every column boundary has a clamp driver, so the formulation holds for any parallel-rail topology whose site condenses to one branch.

## Structural assumptions

The formulation is specialised to a **parallel BL/SL** array — the BL rail and the SL rail run side by side along the row direction — and rests on four assumptions:

1. **Exactly two array rails.** Each site couples a BL node and an SL node; the coupled wire Newton is therefore a block-$2\times2$ per node.
2. **One signed two-terminal branch.** The two rails couple only through a single signed cell branch current; the cell self-condenses any internal node, so the array carries no per-cell internal unknown.
3. **The control line is a driven boundary.** The gate/control line (the word line) is an externally driven boundary, not a solved mesh node. Hence the columns and their driver pairs are mutually independent.
4. **Each rail is a 1-D series ladder.** IR drop accumulates along the row axis of each rail, giving the (block-)tridiagonal structure the Thomas sweep exploits.

The orthogonal case (BL $\perp$ SL forming a 2-D mesh, where the two rails are *not* parallel and a line is a solved node) violates assumption 3 and is out of scope here.

## Formulation

The solve is a nested (block-Gauss-Seidel) decomposition: an *outer* $2\times2$ Newton on the per-column clamp pair $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ wrapped around an *inner* array solve at a frozen clamp pair. The inner solve is a coupled block-$2\times2$ wire Newton on $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ along the rows of each column. The cell does not enter the array unknowns: at each inner step every cell condenses its internal node to a single branch current with its two signed terminal conductances, which enter the wire Newton directly.

## Cell branch and driver transfer

The formulation is defined over two constitutive relations — the condensed cell branch at every array site and the monotone clamp-driver transfer at every column boundary.

The cell branch:

- **Single branch current.** Each site carries one condensed current $I_{\mathrm{cell}}(V_{\mathrm{BL}}, V_{\mathrm{SL}})$, positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$. The same current leaves the BL wire KCL and enters the SL wire KCL, so the array carries no per-cell internal residual.
- **Signed terminal derivatives.** $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}} \le 0$. These definite signs make the inner wire system a well-posed M-matrix-flavour problem (below).

The clamp-driver transfer (one BL driver and one SL driver per column boundary):

- **Monotone scalar transfer.** Each driver maps its boundary port current to a clamp voltage with a strict, definite-sign response: the BL clamp driver strictly monotone in $I_{\mathrm{BL,port}}$, the SL driver strictly monotone in $I_{\mathrm{SL,port}}$.

The formulation holds for any parallel-rail topology whose site condenses to one signed branch and whose boundaries present a monotone scalar transfer. The per-cell condensation and its monotonicity are specified in [cell](../cell/family.md); the clamp-driver transfer function in [voltage driver](../../analog/voltage_driver.md).

## Newton linearization

The Newton steps linearize the residuals analytically over three Jacobians. Write the cell's effective terminal conductances as two non-negative magnitudes,

$$g_{\mathrm{BL,eff}} \equiv \frac{\partial I_{\mathrm{cell}}}{\partial V_{\mathrm{BL}}} \ge 0, \qquad g_{\mathrm{SL,eff}} \equiv -\frac{\partial I_{\mathrm{cell}}}{\partial V_{\mathrm{SL}}} \ge 0,$$

so the SL sign is absorbed into a magnitude and the assembled Jacobians are sign-uniform.

### Inner wire Jacobian

The inner solve is a coupled block-$2\times2$ wire Newton on $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ along the row axis; its residuals are the per-node wire KCL with the shared cell branch entering both rails — the BL residual $F_{\mathrm{BL}}$ takes $+I_{\mathrm{cell}}$ (drained from BL), the SL residual $F_{\mathrm{SL}}$ takes $-I_{\mathrm{cell}}$ (injected into SL). The Jacobian $J_{\mathrm{inner}}$ is block-tridiagonal in the row index $k$, and its per-node diagonal block carries the cell's cross-coupling between the two rails,

$$J^{\mathrm{diag}}_k = \begin{bmatrix} d^{\mathrm{BL}}_k + g_{\mathrm{BL,eff},k} & -\,g_{\mathrm{SL,eff},k} \\ -\,g_{\mathrm{BL,eff},k} & d^{\mathrm{SL}}_k + g_{\mathrm{SL,eff},k} \end{bmatrix},$$

where $d^{\mathrm{BL}}_k$, $d^{\mathrm{SL}}_k$ are the BL / SL wire-ladder diagonal entries at node $k$ and the cross terms are $\partial F_{\mathrm{BL}}/\partial V_{\mathrm{SL}} = -g_{\mathrm{SL,eff}}$ and $\partial F_{\mathrm{SL}}/\partial V_{\mathrm{BL}} = -g_{\mathrm{BL,eff}}$. The sub- and super-diagonal blocks are diagonal $2\times2$, carrying the negated intra-rail wire off-diagonals on their respective rails: BL and SL are independent ladders with no cross-rail wire coupling. For an SL-grounded chip $g_{\mathrm{SL,eff}}$ and the SL wire's contribution to the BL drop are tiny, so the coupling reduces numerically to near-independent BL / SL solves; a variable-SL chip retains the full linearization through the same block.

### Inner-to-clamp sensitivity

The outer Newton needs the response of the port-adjacent node voltages ($k = 0$) to the clamp pair — the $2\times2$ implicit-function-theorem sensitivity

$$K \equiv \frac{\partial V_{\mathrm{node},0}}{\partial V_{\mathrm{clamp}}} = \begin{bmatrix} \partial V_{\mathrm{BL},0}/\partial V_{\mathrm{BL,CL}} & \partial V_{\mathrm{BL},0}/\partial V_{\mathrm{SL,CL}} \\ \partial V_{\mathrm{SL},0}/\partial V_{\mathrm{BL,CL}} & \partial V_{\mathrm{SL},0}/\partial V_{\mathrm{SL,CL}} \end{bmatrix}.$$

At the inner-converged state, differentiating the inner system with respect to a clamp perturbation gives $J_{\mathrm{inner}}\,\mathbf{u} = \mathbf{b}$. A unit $V_{\mathrm{BL,CL}}$ perturbation forces only the BL residual at node 0, so $\mathbf{b} = G^{\mathrm{BL}}_{\mathrm{seg},0}\,\mathbf{e}_0^{\mathrm{BL}}$ with $\mathbf{e}_0^{\mathrm{BL}}$ the node-0 BL basis vector; analogously for $V_{\mathrm{SL,CL}}$. The two columns of $K$ are the node-0 rows of the two basis solves,

$$K_{:,0} = G^{\mathrm{BL}}_{\mathrm{seg},0}\,\big(J_{\mathrm{inner}}^{-1}\,\mathbf{e}_0^{\mathrm{BL}}\big)_{0,:}, \qquad K_{:,1} = G^{\mathrm{SL}}_{\mathrm{seg},0}\,\big(J_{\mathrm{inner}}^{-1}\,\mathbf{e}_0^{\mathrm{SL}}\big)_{0,:},$$

where $G^{\mathrm{BL}}_{\mathrm{seg},0}$, $G^{\mathrm{SL}}_{\mathrm{seg},0}$ are the driver-to-first BL / SL segment conductances. The two basis solves are mathematically independent, each a block-tridiagonal solve against the same coupled $J_{\mathrm{inner}}$.

### Outer clamp Newton

The outer step is a per-column $2\times2$ Newton on the clamp pair with residual $F_{\mathrm{outer}}(V_{\mathrm{clamp}}) = V_{\mathrm{target}}(V_{\mathrm{clamp}}) - V_{\mathrm{clamp}}$, where each clamp driver maps its first-segment port current to a target clamp voltage,

$$V_{\mathrm{BL,target}} = \operatorname{driver}_{\mathrm{BL}}\!\big(G^{\mathrm{BL}}_{\mathrm{seg},0}\,(V_{\mathrm{BL,CL}} - V_{\mathrm{BL},0}(V_{\mathrm{clamp}}))\big), \qquad V_{\mathrm{SL,target}} = \operatorname{driver}_{\mathrm{SL}}\!\big(G^{\mathrm{SL}}_{\mathrm{seg},0}\,(V_{\mathrm{SL,CL}} - V_{\mathrm{SL},0}(V_{\mathrm{clamp}}))\big).$$

Writing $g_{\mathrm{BL}} \equiv G^{\mathrm{BL}}_{\mathrm{seg},0}$ and the driver small-signal slope $r_{\mathrm{BL}} \equiv \partial V_{\mathrm{BL,target}}/\partial I_{\mathrm{BL,port}}$ (likewise for SL), $\partial V_{\mathrm{target}}/\partial V_{\mathrm{clamp}}$ expands as the driver slope times the port-current sensitivity, with $K$ carrying the node-0 response,

$$\frac{\partial F_{\mathrm{outer}}}{\partial V_{\mathrm{clamp}}} = \begin{bmatrix} r_{\mathrm{BL}}\,g_{\mathrm{BL}}\,(1 - K_{00}) - 1 & -\,r_{\mathrm{BL}}\,g_{\mathrm{BL}}\,K_{01} \\ -\,r_{\mathrm{SL}}\,g_{\mathrm{SL}}\,K_{10} & r_{\mathrm{SL}}\,g_{\mathrm{SL}}\,(1 - K_{11}) - 1 \end{bmatrix}.$$

Each outer step solves $\dfrac{\partial F_{\mathrm{outer}}}{\partial V_{\mathrm{clamp}}}\,\delta = -F_{\mathrm{outer}}$ per column for the clamp update $\delta$, damped by a fixed per-step cap.

## Well-posedness

The monotonicity directions follow from the cell's signed-conductance contract and the wire-ladder structure. The cell branch current is increasing in $V_{\mathrm{BL}}$ and decreasing in $V_{\mathrm{SL}}$ ($\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}} \ge 0$, $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}} \le 0$), so the inner coupled wire system is a block-$2\times2$ tridiagonal M-matrix-flavour system with a unique fixed point at any frozen clamp pair. Because the cell branch currents are monotone in the node voltages, the boundary port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ are themselves monotone in the clamp voltages, so each boundary clamp-driver response is strictly monotone in a definite direction: raising $V_{\mathrm{BL,CL}}$ increases the cell read current and hence the BL port current it must absorb, while raising $V_{\mathrm{SL,CL}}$ lowers the cell drive and hence the SL port current. The outer map on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ composes the strictly monotone driver responses (the BL clamp driver strictly monotone in $I_{\mathrm{BL,port}}$, the SL driver strictly monotone in $I_{\mathrm{SL,port}}$) with the strictly monotone array response, giving a unique fixed point; the damped $2\times2$ Newton converges quadratically near it. Rail pseudo-equilibria are excluded, because the outer Newton is a well-conditioned per-column $2\times2$ problem away from the rails; a rail is reached only when the port current is genuinely outside the driver's reachable range, where the rail is the correct physics.

The formulation carries IR drop through the per-segment interconnect resistances of the wire ladder: node voltages along the row axis differ from the clamp voltage by the resistive drop the segment currents develop, and these drops enter the wire-ladder KCL residuals directly, for any per-segment interconnect-resistance profile along the row axis.

## Symbols

Shared electrical symbols are pinned in [notation_conventions](../../../../conventions/notation_conventions.md); the per-call cell branch quantities are defined in [cell](../cell/family.md).

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | BL wire node voltage | V | `v_bl_node` |
| $V_{\mathrm{SL}}$ | SL wire node voltage | V | `v_sl_node` |
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_bl_clamp` |
| $V_{\mathrm{SL,CL}}$ | SL clamp voltage | V | `v_sl_drive` |
| $I_{\mathrm{cell}}$ | condensed cell branch current (BL $\to$ SL) | uA | `cell.solve_branch` |
| $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `cell.solve_branch` |
| $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `cell.solve_branch` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | derived from node voltages |
| $I_{\mathrm{SL,port}}$ | SL boundary port current | uA | derived from node voltages |

## Validation

TODO: add validation evidence for solver fixed points, converged residuals, and per-cell finite-difference device derivatives.

## References

TODO: cite the Newton / block-tridiagonal solution methods.

---

- **Internals**: [solver internals](../../../../internals/primitive/xbar/solver.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../../api/README.md) (`[cim_macro.array_config.solver_config]`)
