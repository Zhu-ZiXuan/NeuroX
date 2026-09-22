# CIM unit

A CIM unit realizes the [integer operator](family.md) through finite-capacity [macro reads](../../primitive/macro/cim/family.md) and digital recovery. Precision slicing represents values in the macro's input and weight domains; geometric placement assigns those slices to physical instances and temporal reuse slots. Recovery combines the resulting codes into logical outputs.

## Precision slicing

A value $v$ is represented by $S$ slices with positive positional weights:

$$v=\sum_{s=0}^{S-1}d_s p_s,\qquad p_s=R^s.$$

Multiple slices require an explicit unsigned, true-form, or canonical encoding. The slice count is fixed; encoding adds no correction slices. Unsigned slices occupy $[0,R-1]$; true-form and canonical slices occupy $[-(R-1),R-1]$. Signed slice values cannot be replaced by fixed positional signs. Complement encoding is outside the slicing model.

Slicing selects the largest radix, bounded by the carrier's positive maximum plus one, whose digit intervals fit the carrier. Input slicing uses the native input carrier, so signed encodings require support for signed inputs. Weight slicing can obtain a signed carrier through paired nonnegative outputs. Decomposition and recovery use the same positive positional weights. Values outside the representable range may wrap while emitted slices remain within their bounds.

A direct single slice retains the original value with positional weight one and requires no encoding or positional recovery circuit. True-form or canonical weight selection can still request a signed range on a nonnegative macro through weight-polarity mapping.

Weight slices are concatenated in slice-major order before tiling, expanding $N$ logical outputs to $S_wN$ independent slice outputs. Tile boundaries may cross slice boundaries, and slices of one logical output may occupy different tiles or input slots. Input slices reuse the programmed weights in successive reads: increasing $S_x$ increases access count without increasing macro population.

## Weight polarity

A nonnegative weight carrier $[0,U]$ represents signed slices in $[-U,U]$ using adjacent positive and negative outputs within one macro:

$$w_s^+=\max(w_s,0),\qquad w_s^-=\max(-w_s,0),\qquad w_s=w_s^+-w_s^-.$$

For each input position, the physical output order is $w_0^+,w_0^-,w_1^+,w_1^-,\ldots$. A macro with $O$ physical outputs holds $Q=\lfloor O/2\rfloor$ logical slice outputs. Placement reserves complete pairs within one macro and input slot; an odd final physical port remains zero and disabled. At least two physical outputs are required. An access with $n$ valid logical slice outputs enables its first $2n$ physical outputs, including both members of a numerically zero pair.

Physical outputs visit every lane of a scan before advancing to the next scan. With one lane, a pair arrives in consecutive scans; with an even lane count, both terms arrive in one scan. An odd lane count may split a pair across scans. The local stage retains the positive code until its negative partner arrives, then forms their difference before phase and input-slice recovery.

Each polarity retains its own conversion. In general, $\mathcal{Q}(a^+)-\mathcal{Q}(a^-)\ne\mathcal{Q}(a^+-a^-)$. A configured two-input adder adds the positive code to the negated negative code, wraps at its output width, and bills one operation per enabled pair. An unconfigured difference uses exact subtraction without circuit cost.

For $L_r$ readout lanes, at most $L_d=\min(Q,\lceil L_r/2\rceil)$ pairs complete in one scan. Each macro has $L_d$ local accumulators and, when configured, $L_d$ difference circuits, reused across scans with routing and buffering. Their throughput is assumed sufficient for the scan schedule; partial-sum storage belongs to the unit-local peripheral budget. Global recovery receives one signed result per pair.

Native signed carriers and encodings that emit only nonnegative slices use the full output capacity without polarity expansion. True-form and canonical encodings require pairs on an unsigned macro. Pairing also applies to a direct single weight slice; true-form magnitude slicing over $S_w$ slices accepts $[-((U+1)^{S_w}-1),(U+1)^{S_w}-1]$.

## Tiling and input-slot sharing

Let $K$ be the contraction width, $I$ the macro input capacity, and $Q$ the logical slice-output capacity: $O$ for native mapping or $\lfloor O/2\rfloor$ for differential mapping. Geometric placement partitions the contraction and expanded output dimensions without decomposing values. Tile input width $L$, contraction-tile count $T_c$, output-block count $B$, and per-macro input-slot capacity $C$ are

$$L=\min(K,I),\qquad T_c=\left\lceil\frac{K}{L}\right\rceil,\qquad B=\left\lceil\frac{S_wN}{Q}\right\rceil,\qquad C=\left\lfloor\frac{I}{L}\right\rfloor.$$

Input-slot sharing assigns those output blocks to $G$ macro groups and $D$ serial reuse steps:

$$G=\left\lceil\frac{B}{C}\right\rceil,\qquad D=\left\lceil\frac{B}{G}\right\rceil.$$

Output block $b=dG+g$ occupies input slot $[dL,(d+1)L)$ of macro group $g$. Without sharing, $C=1$, $G=B$, and $D=1$. Linear operators use no input-slot sharing; convolution operators may enable it. Sharing changes physical multiplicity and duration while preserving the logical matrix operation.

Only the ends of the contraction and expanded output dimensions are padded. Missing final blocks and unused ports are programmed to zero. Each scheduled block carries its valid physical output count, including zero for absent blocks and both outputs of a differential pair. Valid zero-valued weights still count. The same placement-derived widths govern electrical execution, digital gating, and timing; different reuse slots may have different valid widths.

## Input phases

The simultaneous-input limit $A$ divides a tile's $L$ input positions into the minimum number of phases:

$$P=\left\lceil\frac{L}{A}\right\rceil,\qquad q=\left\lfloor\frac{L}{P}\right\rfloor,\qquad r=L\bmod P.$$

The first $r$ phases receive $q+1$ consecutive positions and the remaining phases receive $q$. Group sizes differ by at most one and never exceed $A$. Each phase retains selected positions at their original input indices and masks the rest to zero, preserving weight placement. Macro conversion runs separately for every phase before its output codes are accumulated.

## Recovery and circuit costs

After macro reads, recovery proceeds in this order:

1. Recover each positive/negative pair locally, when polarity mapping is required.
2. Sum input phases for each input slice, then combine input slices with their positional weights. Weight slices and input tiles retain independent partial sums.
3. Restore output-tile ordering from the input slots without summing distinct output tiles.
4. Accumulate contraction-tile contributions at each output-tile port.
5. Concatenate output tiles, remove padding, and restore separate weight-slice results.
6. Reconstruct the weight slices once for each logical output.

Each native lane or simultaneously completed differential pair has one local radix accumulator shared by phase and input-slice recovery. Phase recovery uses radix one. Both stages apply the circuit's register width and bill enabled updates, including numerical zeros, while sharing static hardware costs. Scan positions retain separate partial sums while reusing that circuit.

Global contraction-tile recovery has $BQ$ accumulators, including padded output ports. Final weight reconstruction has $N$ radix summators when configured for multiple slices. Each arithmetic stage retains its own register-width and energy boundary under the [digital circuit model](../../primitive/digital.md). Unconfigured recovery uses exact integer arithmetic without circuit cost; direct single slices bypass positional recovery.

Invalid ports are disabled during local and contraction-tile accumulation, incur no evaluation energy, and are removed before weight reconstruction. Ordering and trimming have no separately modeled circuit cost.

## Timing

### Local accumulation

Input slices, input phases, and input-slot selections execute serially; contraction tiles and macro groups execute in parallel. Serial accesses wait for their slowest active macro. Local digital updates overlap subsequent analog scans, and a complete access waits for its final update before the next access starts.

With sufficient digital throughput, a nonempty macro access including local recovery takes

$$t_{\mathrm{access}}=t_{\mathrm{macro}}+t_{\mathrm{clk}}.$$

The macro duration includes every scan needed to read both physical outputs of each valid pair. Polarity recovery and local accumulation complete in the same digital cycle, retaining separate arithmetic and energy costs. Configuring a polarity adder adds no extra cycle. An access with no valid outputs takes zero time. Completing all serial accesses gives one input vector's local duration $t_{\mathrm{local}}$.

### Input pipeline

Global collection transfers one contraction tile's complete output row per clock period, including all weight slices and output-tile ports after polarity recovery. For $T_c$ tiles, the first row arrives one period after local completion and the last after $T_c$ periods. Each row occupies one subsequent accumulation period. Final weight reconstruction adds $C_w=1$ cycle when configured, or $C_w=0$ for direct or unmodeled recovery:

$$t_{\mathrm{global}}=(T_c+1+C_w)t_{\mathrm{clk}}.$$

One linear VMM completes after local execution and global recovery:

$$t_{\mathrm{VMM}}=t_{\mathrm{local}}+t_{\mathrm{global}}.$$

Convolution windows reuse the programmed matrix. Result buffering overlaps the next window's local work with the preceding window's global transfer and recovery. For $M=H_{\mathrm{out}}W_{\mathrm{out}}$ windows, one complete image takes

$$t_{\mathrm{image}}=t_{\mathrm{local}}+t_{\mathrm{global}}+(M-1)\max(t_{\mathrm{local}},T_c t_{\mathrm{clk}}).$$

Independent convolution groups apply the mapping separately and execute on separate circuits in parallel, under the [grouped convolution model](conv2d.md). Batch, token, and timestep counts enumerate basic operations without multiplying their individual durations; the task schedule composes those operations at each retained observation position.

## Assumptions and validity

Mapping uses integer arithmetic. Its storage width must represent each operation's values and intermediates independently of modeled register widths. Analog and conversion effects arise inside macro accesses; ideal accesses provide a control for the same mapping and recovery.

Timing assumes sufficient lane throughput, full-row transfer bandwidth, and result buffering. It omits clock-edge alignment and digital backpressure. Padding represents executed work unless its activity is explicitly disabled. Digital PPA characterizes each implementation at its selected width and the common clock period; changing that period supplies no automatic PPA scaling law. Analog durations are not rounded to clock edges.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $N,K$ | logical output and contraction widths | — | matrix dimensions |
| $v,d_s$ | original integer and slice value at index $s$ | — | slicing input and output |
| $s,S$ | slice index and fixed slice count | — | slice axis, `slice_num` |
| $R,p_s$ | slice radix and positive positional weight | — | `slice_radix`, `place_values` |
| $S_w,S_x$ | weight- and input-slice counts | — | `w_slice_num`, `x_slice_num` |
| $U$ | nonnegative weight-carrier maximum | — | upper weight bound |
| $w_s,w_s^+,w_s^-$ | signed slice and its nonnegative polarity terms | — | logical and programmed slice values |
| $a^+,a^-$ | analog outputs of the two polarity branches | — | pre-conversion branch values |
| $\mathcal{Q}$ | analog-to-code quantization | — | per-output conversion |
| $I,O,A$ | macro input capacity, physical output capacity, and simultaneous-input limit | — | `input_num`, `output_num`, `max_active_num` |
| $L,Q$ | tile input and logical slice-output capacities | — | `tiler.input_per_tile`, `tiler.output_per_tile` |
| $L_r,L_d$ | readout lanes and maximum pairs completed per scan | — | `lane_num`, `local_lane_num` |
| $n$ | valid logical slice outputs in one access | — | valid width before polarity expansion |
| $T_c$ | contraction-tile count | — | `tiler.input_tile_num` |
| $B,C$ | output-block count and per-macro input-slot capacity | — | `tiler.output_tile_num`, `merge.input_slot_capacity` |
| $G,D$ | macro-group and serial input-slot step counts | — | `merge.macro_group_num`, `merge.merge_step_num` |
| $b,d,g$ | output-block, input-slot, and macro-group indices | — | — |
| $P,q,r$ | phase count, minimum phase size, and number of larger phases | — | input-phase partition |
| $t_{\mathrm{access}}$ | macro access including the final local update | ns | — |
| $t_{\mathrm{macro}}$ | complete analog macro VMM duration | ns | macro `latency__ns` |
| $t_{\mathrm{clk}}$ | common digital clock period | ns | `clock_period__ns` |
| $t_{\mathrm{local}},t_{\mathrm{global}}$ | local access and global recovery durations | ns | — |
| $C_w$ | final weight-reconstruction cycles | — | `w_radix_summator_config` |
| $M,H_{\mathrm{out}},W_{\mathrm{out}}$ | convolution-window count and output spatial extents | — | convolution output geometry |
| $t_{\mathrm{VMM}},t_{\mathrm{image}}$ | complete linear VMM and convolution image durations | ns | unit `latency__ns` |

## Validation

`tests/architecture/unit/cim/test_execution.py` checks numerical recovery, physical placement, quantization and register-width ordering, and circuit costs. `tests/architecture/unit/cim/test_latency.py` checks local scheduling. Operator tests cover complete VMM and convolution execution, including grouped geometry; mapping-tool tests check their individual transforms.
