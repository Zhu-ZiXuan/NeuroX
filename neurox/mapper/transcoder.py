"""Data-representation transcoders used by the xbar mappers.

The xbar-macro mapping layer is split into a unified
:class:`neurox.mapper.xbar.XbarMapper` (composed of a tiler plus an
activation slicer + a weight slicer).  Each slicer composes one or
more :class:`SignedDigitTranscoder` instances at call time; this
file owns the underlying digit encode / decode math and the base
``value_range`` of a single positional-radix digit string.

A ``Transcoder`` converts an integer tensor into a per-digit signed
representation suitable for programming individual crossbar cells.
The slicer inserts the digit axis at the appropriate position
before the mapper tiles the tensor into the physical layout.

Supported encodings (selected by string literal at construction time):

``"true_form"``
    Sign-magnitude decomposition.  The magnitude is split into
    ``digit_num`` base-``radix`` digits; each is multiplied by the
    original sign.  Digits lie in ``[-(r-1), r-1]``.

``"complement"``
    Radix-``r`` complement.  Lower digits are extracted via successive
    ``% radix`` / ``// radix`` steps; the MSB is sign-corrected in
    place so every digit decodes with a positive ``radix**i`` weight.
    Digits lie in ``[0, r-1]`` except the MSB which is in
    ``[-(r-1), r-1]``.

``"canonical"``
    Canonical signed digit (non-adjacent form) encoding that minimises
    the non-zero digit count via carry propagation.  Digits lie in
    ``[-(r//2), r//2]``.

Digit ordering: LSB first (``digits[..., 0, ...]`` is always the
least-significant digit).

Decoding: :meth:`Transcoder.decode` recovers the original integer from
any scheme by computing the positional weighted sum
``sum_i(digit[i] * radix**i)``.
"""

from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

import torch
from torch import Tensor


class Transcoder(ABC):
    """Base class for data-representation converters.

    Subclasses must implement ``encode``, ``decode``, ``value_range``,
    and expose ``digit_num`` (the size of the axis inserted by
    ``encode``) plus ``radix`` (the positional base).
    """

    @abstractmethod
    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode ``x`` into the target representation.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Encoded tensor with a new size-``digit_num`` axis at ``dim``.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, digits: Tensor, *, dim: int = -1) -> Tensor:
        """Decode a digit tensor back to integers.

        Args:
            digits: Digit tensor produced by ``encode``.
            dim: Axis of the digit dimension to reduce.

        Returns:
            Integer tensor with ``dim`` removed.
        """
        raise NotImplementedError

    @abstractmethod
    def value_range(self) -> tuple[int, int]:
        """Inclusive integer range a single digit string can represent.

        Returns the algorithm-side ``(lo, hi)`` envelope of the
        ``(encoding, radix, digit_num)`` triple — the range any
        scalar consumed by :meth:`encode` is expected to lie in and
        :meth:`decode` is guaranteed to reproduce.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def digit_num(self) -> int:
        """Number of digits produced by ``encode``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def radix(self) -> int:
        """Positional radix ``r`` of the digit representation.

        Each digit position ``i`` carries weight ``r**i`` when
        :meth:`decode` reconstructs the integer from its digits.
        """
        raise NotImplementedError


Encoding: TypeAlias = Literal["true_form", "complement", "canonical"]


class SignedDigitTranscoder(Transcoder):
    """Transcoder converting 2's-complement integers to signed-digit form.

    Each instance is bound to a fixed ``(encoding, radix, digit_num)``
    triple so callers pass only the tensor and axis.

    Args:
        encoding: ``"true_form"``, ``"complement"``, or ``"canonical"``.
        radix: Radix base ``r``; each digit lies in ``[-(r-1), r-1]``.
        digit_num: Number of output digits (size of the inserted axis).
    """

    def __init__(self, encoding: Encoding, radix: int, digit_num: int) -> None:
        self._encoding = encoding
        self._radix = radix
        self._digit_num = digit_num

    @property
    def encoding(self) -> Encoding:
        return self._encoding

    @property
    def radix(self) -> int:
        return self._radix

    @property
    def digit_num(self) -> int:
        return self._digit_num

    # --- public interface --- #

    def encode(self, x: Tensor, *, dim: int = -1) -> Tensor:
        """Encode integer tensor into signed-digit form.

        Args:
            x: Integer tensor to encode.
            dim: Axis at which the digit dimension is inserted.

        Returns:
            Signed-digit tensor with a new size-``digit_num`` axis at ``dim``.
        """
        match self._encoding:
            case "true_form":
                return self._encode_true_form(x, dim)
            case "complement":
                return self._encode_complement(x, dim)
            case "canonical":
                return self._encode_canonical(x, dim)

    def decode(self, digits: Tensor, *, dim: int = -1) -> Tensor:
        """Decode a signed-digit tensor back to integers.

        Computes ``sum_i(digit[i] * radix**i)`` along ``dim``.

        Args:
            digits: Signed-digit tensor.
            dim: Axis of the digit dimension to reduce.

        Returns:
            Integer tensor with ``dim`` removed.
        """
        scales = torch.tensor(
            [self._radix**i for i in range(digits.size(dim))],
            device=digits.device,
            dtype=digits.dtype,
        )
        shape = [1] * digits.ndim
        shape[dim] = digits.size(dim)
        return (digits * scales.view(*shape)).sum(dim=dim)

    def value_range(self) -> tuple[int, int]:
        """Inclusive integer range one digit string of this transcoder represents.

        All three encodings reconstruct the same algorithm-side
        envelope ``[-(r^D - 1), r^D - 1]`` where ``r = radix`` and
        ``D = digit_num`` — ``true_form`` covers it symmetrically by
        construction, and ``complement`` / ``canonical`` reach the
        same magnitude through the sign-corrected MSB.  Single-
        positional-radix digit strings always have a value_range
        symmetric about zero, so the slicers above this layer can
        rely on that envelope without caring which encoding is in
        use.
        """
        n_max = self._radix**self._digit_num - 1
        return -n_max, n_max

    # --- encoding implementations --- #

    def _encode_true_form(self, x: Tensor, dim: int) -> Tensor:
        sign = x.sign()
        x = x.abs()
        all_digits: list[Tensor] = []
        for _ in range(self._digit_num):
            rem = x % self._radix
            x = x // self._radix
            all_digits.append(rem * sign)
        return torch.stack(all_digits, dim=dim)

    def _encode_complement(self, x: Tensor, dim: int) -> Tensor:
        radix = self._radix
        all_digits: list[Tensor] = []
        for _ in range(self._digit_num):
            rem = x % radix
            x = x // radix
            all_digits.append(rem)
        msb = all_digits[-1]
        all_digits[-1] = torch.where(msb >= (radix + 1) // 2, msb - radix, msb)
        return torch.stack(all_digits, dim=dim)

    def _encode_canonical(self, x: Tensor, dim: int) -> Tensor:
        radix = self._radix
        all_digits: list[Tensor] = []
        for _ in range(self._digit_num):
            rem = x % radix
            x = x // radix
            mod = x % radix
            carry = rem > (radix // 2)
            carry = carry & (mod != 0)
            carry = carry | (mod == (radix - 1))
            carry = carry & (rem != 0)
            x = x + carry
            all_digits.append(torch.where(carry, rem - radix, rem))
        return torch.stack(all_digits, dim=dim)
