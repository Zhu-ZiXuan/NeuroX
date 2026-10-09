# Record collection

Components emit observations; execution transports them across supported loops; recorders retain completed snapshots. This separation keeps energy and diagnostic data out of numerical callback return values without changing the meaning of physical accesses.

## Supported boundaries

Only NeuroX's `run_scan` family and `run_map` provide hidden record outputs. They mark their own scopes explicitly. The adapters in `common.torch_compat` retain their original behavior; raw PyTorch loops are not implicitly enhanced.

A standalone recorded loop enters the scoped compiler automatically. An enclosing compiled application must use `compile_recording` or select `recording_backend` in `torch.compile`. Project-owned compiled entry points use this backend. Ordinary eager submissions continue to work directly.

The compiler captures registered recorder operations, snapshots their tensor payloads at the emission point, and adds those payloads to the supported loop's outputs. Native scan/map output collection stores the payloads. Numerical carry and public callback results remain unchanged. Outside the compiled numerical region, an execution wrapper delivers the completed observations to their original recorders and returns only the original numerical outputs.

Record identities, payload structure and per-step tensor metadata are fixed within one captured invocation. Tensor values and recorder identities remain runtime inputs. A disabled recorder does not cause an observation to be computed or collected merely because a loop supports recording.

Hidden records cannot cross raw scan/map, while-loop or conditional boundaries. A supported scan nested under an unsupported boundary does not bypass that restriction. Such a submission is rejected during compilation. Numerical control flow without submissions retains its ordinary semantics.

## Record layout

Collection does not infer whether an axis denotes physical reuse, an independent sample or a diagnostic iteration.

A scan's `record_dim` places its added record axis. By default the axis follows the scan output layout; `-1` appends an iteration axis. Reverse scans restore record positions to input order alongside their numerical outputs. Maps retain their leading mapped axis.

`run_chunked` owns positional reconstruction. Its hidden outputs follow the same balanced groups, flattening, concatenation and `leading_shape` restoration as numerical results. Energy observation-prefix reduction happens after this reconstruction. Submissions inside a chunk must retain that chunk's position prefix; an already reduced scalar cannot identify which original observations contributed to it.

Array accounting declares two additional transformations in its compiled owner. The phase scope sums only additive energy over the declared WL phase axes. The access scope combines phase and rest contributions with the same identity into one completed-access submission. These transformations preserve the root profiler's separate `on_repeat` policy, including `replace`. Rest energy is established once per held access, not once per chunk or numerical iteration.

## Energy and diagnostics

Energy uses the emitter's retained observation prefix. Trailing hardware and work axes are reduced after their execution layouts are restored. Source names, channels and numerical dtypes are retained.

`TraceCollector` retains separate diagnostic snapshots. Its fields keep their positional and history axes, dtypes and invalid-value sentinels. `SolvingTrace.record` flattens nested field names into dotted paths without changing the returned history. Array `solve_dc_trace` also records its completed trace when a collector is active; the array must have been named with `stamp_names`.

A collector does not enable solver tracing, select different stopping tolerances or permit an unconverged result. Ordinary `solve_dc` remains strict. `solve_dc_trace` keeps its existing diagnostic contract and explicit return value, so a compiled caller can still consume that history directly. Observations generated inside an unsupported conditional must remain ordinary branch outputs until they reach a supported recording boundary.

## Lifetime and resource use

A compiled invocation delivers its records after its numerical computation returns successfully. Nested loops transmit hidden tensors to that boundary rather than calling Python recorders at every iteration. Recorder contexts own the final commit: existing failed-context behavior discards records, and context exit waits for asynchronous CPU exports. An execution failure that prevents the compiled region from returning cannot expose its unfinished output pools.

The pool stores one tensor per static submission site and iteration. Its storage therefore grows with iteration count and retained observation size. Energy should be normalized to a compact observation layout as soon as its positional transformations permit. Nested trace histories retain all requested history axes and can be much larger than energy. A mask that fills unselected entries with invalid values does not reduce storage; field or position selection requires an explicit diagnostic policy.

The supported collection path is inference/no-grad with fixed record schemas. Version checks cover the minimum supported PyTorch version and current development CPU/CUDA execution. Higher-order operators and graph metadata remain version-sensitive even though the raw PyTorch adapters are unchanged.
