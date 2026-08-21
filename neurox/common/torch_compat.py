"""Type-preserving boundaries for untyped PyTorch callables."""

from collections.abc import Callable
from typing import Protocol, overload

import torch


class _Decorator(Protocol):
    def __call__[**P, ResultT](self, fn: Callable[P, ResultT]) -> Callable[P, ResultT]: ...


class _CompilerDisable(Protocol):
    @overload
    def __call__[**P, ResultT](
        self,
        fn: Callable[P, ResultT],
        recursive: bool = True,
        *,
        reason: str | None = None,
    ) -> Callable[P, ResultT]: ...

    @overload
    def __call__(
        self,
        fn: None = None,
        recursive: bool = True,
        *,
        reason: str | None = None,
    ) -> _Decorator: ...


torch_compiler_disable: _CompilerDisable = torch.compiler.disable
