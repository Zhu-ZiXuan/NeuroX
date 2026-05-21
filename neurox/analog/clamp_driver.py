"""Port clamp-driver protocol."""

from typing import Protocol, runtime_checkable

from torch import Tensor

# ---------------------------------------------------------------------------
# Clamp-driver Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ClampDriver(Protocol):
    """Protocol surface of voltage-clamped driver."""

    @property
    def v_ref__V(self) -> float: ...

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snapshot: object,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...
