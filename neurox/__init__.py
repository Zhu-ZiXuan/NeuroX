"""NeuroX: PyTorch-based neuromorphic computing simulation framework.

The top-level surface is organised around three user-visible stages:

1. **Preparation / training** — replace float ops with HAT operators,
   fold BatchNorm, freeze observers, export a NeuroX-flat checkpoint::

       neurox.replace_for_hat(model, macro_factory, spec)
       # ... train ...
       neurox.freeze_hat_observers(model)
       state = neurox.extract_neurox_state(model)

2. **Evaluator construction** — bind the checkpoint to a hardware
   backend with the one-shot wrapper, or use the staged functions for
   selective / heterogeneous flows::

       model = neurox.build_evaluator(model, checkpoint, macro_factory)
       # or:
       neurox.replace_model(model, macro_factory, policy=...)
       neurox.load_neurox_state(model, checkpoint)
       neurox.bind_output_calibration(model)
       neurox.fabricate_model(model)

3. **Execution + profiling** — run normal ``forward`` and collect
   hardware metrics through the side-channel logger::

       with neurox.NeuroxProfiler() as profiler:
           logits = model(images)
       print(profiler.summary(static=neurox.NeuroxProfiler.analyze_static(model)))

All other internal APIs (low-level analog / device modules, mapper,
xbar internals) remain in their submodules and are not re-exported.
"""

import torch._dynamo

# Macro-level ``@torch.compile`` boundary settings.  Both flags target
# the gap between dynamo's capture rules and the macro's Python-side
# control flow:
#
# - ``capture_scalar_outputs``: lets the iterative circuit solver's
#   convergence check (``.item() < ATOL``) survive tracing instead of
#   aborting the compile.
# - ``suppress_errors``: when dynamo hits an unsupported pattern (e.g.
#   side-channel profiler-side ``.item()`` from a ``torch.compiler.disable``
#   region) it falls back to eager for that call instead of crashing.
#   Macros that compile cleanly still get the speedup; macros that
#   don't degrade to eager — never to incorrect output.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True

# --- Stage 1: preparation / training-side conversion ---
from neurox.operator import QuantSpec
from neurox.operator.train import (
    extract_neurox_state,
    fold_batchnorm,
    freeze_hat_observers,
    replace_for_hat,
)

# --- Stage 3: execution + profiling ---
from neurox.profiler import (
    NeuroxProfiler,
    ProfiledModule,
    ProfilerReport,
    RuntimeEvent,
    StaticMetrics,
    StaticRecord,
)

# --- Stage 2: evaluator construction ---
from neurox.replace import (
    NeuroxStateError,
    ReplacementContext,
    ReplacementPolicy,
    ReplacementRule,
    StateBindingReport,
    StructuralReport,
    bind_output_calibration,
    build_evaluator,
    by_attr_match,
    default_policy,
    fabricate_model,
    heterogeneous_macro_policy,
    load_neurox_state,
    name_excluded_match,
    replace_model,
)

__all__ = [
    # --- Stage 1 ---
    "QuantSpec",
    "extract_neurox_state",
    "fold_batchnorm",
    "freeze_hat_observers",
    "replace_for_hat",
    # --- Stage 2 ---
    "NeuroxStateError",
    "ReplacementContext",
    "ReplacementPolicy",
    "ReplacementRule",
    "StateBindingReport",
    "StructuralReport",
    "bind_output_calibration",
    "build_evaluator",
    "by_attr_match",
    "default_policy",
    "fabricate_model",
    "heterogeneous_macro_policy",
    "load_neurox_state",
    "name_excluded_match",
    "replace_model",
    # --- Stage 3 ---
    "NeuroxProfiler",
    "ProfiledModule",
    "ProfilerReport",
    "RuntimeEvent",
    "StaticMetrics",
    "StaticRecord",
]
