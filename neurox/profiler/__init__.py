"""Hardware profiler subpackage.

Re-exports the user-facing types so callers can write
``from neurox.profiler import NeuroxProfiler`` without reaching into the
inner modules.  The side-channel mixin :class:`ProfiledModule` is also
exported here so physical modules can import it from a single root.
"""

from .profiled_module import ProfiledModule
from .profiler import (
    NeuroxProfiler,
    ProfilerReport,
    RuntimeEvent,
    StaticMetrics,
    StaticRecord,
)

__all__ = [
    "NeuroxProfiler",
    "ProfiledModule",
    "ProfilerReport",
    "RuntimeEvent",
    "StaticMetrics",
    "StaticRecord",
]
