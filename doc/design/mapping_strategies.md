# Mapping Architecture — Tiler + Slicer + XbarMapper

## Layer responsibilities

NeuroX's hardware-execution path splits mapping across four
sub-layers, with strict one-way coupling — no two adjacent
layers know each other's internals:

```
operator -> macro
              -> mapper  (XbarMapper)
                   ├── tiler   (Tiler)
                   ├── x_slicer (Slicer)
                   └── w_slicer (Slicer)
              -> xbar primitive
```

| Layer        | Owns                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------ |
| `operator`   | Float ↔ int quantization, calls macro, dequantizes output.                                       |
| `macro`      | Orchestration; holds xbar + mapper + digital aggregation; **dispatches** xbar capability at runtime. |
| `mapper`     | Owner of complete mapping semantics; composes tiler + two slicers + macro-canonical reshape.     |
| `tiler`      | Matrix tiling (`N` / `K` split, padding); value-domain agnostic.                                 |
| `slicer`     | Value-domain decomposition; output ``[..., slice_num, digit_num]``.                              |
| `transcoder` | Pure-math digit encode / decode + base ``value_range``.                                          |
| `xbar`       | Per-tile analog VMM + ADC; exposes only primitive capabilities.                                  |

## Decoupling contract

The **single most important** rule: **no two classes are coupled by
ad-hoc parameter subsets, and no cross-class call uses ``**kwargs``
variadics or ``**dict`` splat patterns.**  Every cross-layer call
documents its full argument list inline at the call site and is
mediated by one of:

1. The mapper's six runtime methods (`x_value_range`,
   `w_value_range`, `x_slice_radix`, `w_slice_radix`, `map_x`,
   `map_w`).  Each takes the **same full xbar capability set** as
   keyword-only arguments.  A concrete mapper may ignore
   parameters it does not consume (using `del`); the macro never
   selects a subset.
2. The slicer's unified 3-kwarg runtime contract — every method
   takes `(digit_count, digit_radix, digit_range)`, no more, no
   less, no ``**kwargs``.  `value_range` is the slicer's *output*
   (carried on `SlicingResult` and queried via `value_range()`),
   never an input.  Concrete slicers ignore parameters they don't
   consume but the signature is fixed.
3. The tiler's split plan factories: `make_w_plan(*, n, k,
   col_num, row_num)` and `make_x_plan(*, k, row_num)`.  Neither
   path uses placeholder values from the other.
4. The slicer's uniform `SlicingResult`:
   `(values, slice_weights, digit_weights, value_range)` with
   trailing-2 `[slice_num, digit_num]`.
5. The transcoder's `value_range()` envelope, consumed by the
   slicers.  Slicers never hardcode digit-string range math.

Consequence: each leaf class has exactly one consumer that knows
its surface, and each composite (mapper, macro) sees its children
through abstract base classes.  Adding a new slicer or mapper
strategy never touches its consumers.

## Ownership split

* **mapper** owns the tiler + slicer instances and every static
  strategy parameter (slice counts, encoding choices).
* **macro** owns the xbar and the mapper.  Per call it snapshots
  the xbar's primitive capabilities to **local variables** and
  passes the **full** set as explicit keyword arguments into
  every mapper call.  No helper dict; no splat.
* **xbar** holds primitive capabilities and physical execution;
  never knows about tiling, slicing, or strategy.

## Module layout

```
neurox/mapper/xbar/
├── __init__.py
├── base.py            # XbarMapper, XMappingResult, WMappingResult
├── simple_mapper.py   # SimpleMapper
├── tiler/
│   ├── __init__.py
│   ├── base.py        # Tiler, TilePlan
│   └── simple.py      # SimpleTiler
└── slicer/
    ├── __init__.py
    ├── base.py        # Slicer, SlicingResult
    ├── serial.py      # SerialSlicer
    └── simple.py      # SimpleSlicer
```

The parent package `neurox.mapper` exposes only the shared digit
encoder (`Transcoder` / `SignedDigitTranscoder`); the xbar
mappers and their sub-components are accessed via
`from neurox.mapper.xbar import …`.

## Transcoder contract

`SignedDigitTranscoder(encoding, radix, digit_num)` is a pure
math tool.  It provides:

* `encode(x, *, dim)` → insert a length-`digit_num` axis of
  signed digits at `dim`.
* `decode(digits, *, dim)` → collapse that axis back to a single
  integer.
* `value_range() -> (lo, hi)` → the symmetric envelope
  `(-(r^D - 1), r^D - 1)` one such digit string can represent
  (identical for all three supported encodings — `true_form` is
  symmetric by construction, `complement` / `canonical` reach
  the same magnitude through the sign-corrected MSB).

This is the single source of truth for digit-string range math;
slicers compose transcoder instances and re-export their
`value_range()` envelopes — they never hardcode the formulas.

## Slicer contract

A `Slicer` decomposes one integer scalar into per-slice +
per-digit components.  Output shape: `[..., slice_num, digit_num]`
(trailing-2).  `SlicingResult` carries
`(values, slice_weights, digit_weights, value_range)`.

Runtime contract — every concrete slicer implements:

```python
class Slicer(ABC):
    def value_range(self, *, digit_count, digit_radix,
                    digit_range) -> tuple[int, int]: ...
    def slice_radix(self, *, digit_count, digit_radix,
                    digit_range) -> int: ...
    def slice(self, x, *, digit_count, digit_radix,
              digit_range) -> SlicingResult: ...
```

The three kwargs are mandatory — a slicer ignores those it does
not consume (using `del`) but never widens or narrows the
signature.  `value_range` is the slicer's *output*, carried on
`SlicingResult` and queried via `value_range()`; it is **never** a
slicer input.  The fixed 3-kwarg shape is what makes any concrete
slicer drop-in-replaceable from the mapper's point of view.

Concrete impls:

* **`SerialSlicer(slice_num, encoding)`** — radix-`r` serial
  decomposition with structural `digit_num = 1`.  Use case:
  activation path.  Requires `digit_count == 1`, an unsigned
  `digit_range` (the xbar input grid), and `digit_radix ==
  len(digit_range)`.  Publishes the non-negative half
  `(0, r^Sa - 1)` of the transcoder's symmetric envelope.

* **`SimpleSlicer(slice_num, encoding)`** — direct digitise-then-group.
  Use case: weight path.  It encodes the original integer in one pass
  into `slice_num * digit_count` radix-`digit_radix` digits, then
  reshapes the trailing digit axis into `[slice_num, digit_count]`.
  No outer/inner two-stage transcoding remains.  The published
  `value_range` is therefore the envelope of:
  `SignedDigitTranscoder(encoding, digit_radix, slice_num * digit_count)`.
  The runtime `digit_range` kwarg is kept only because all slicers share
  the same 3-kwarg contract; this concrete slicer does not consume it.

## Tiler contract

A `Tiler` chops a logical `[N, K]` weight matrix into xbar tiles
of size `(data_num, row_num)` and chops a logical `[M, K]`
activation matrix along the shared `K`.  Tiling is
value-domain-agnostic — it operates on tensors whose trailing-2
`[slice_num, digit_num]` already came from the slicer.

The plan factory is split:

```python
w_plan = tiler.make_w_plan(n=N, k=K, col_num=col_num, row_num=row_num)
x_plan = tiler.make_x_plan(k=K, row_num=row_num)

w_tiled = tiler.tile_w(w_sliced, plan=w_plan)
x_tiled = tiler.tile_x(x_sliced, plan=x_plan)
```

Activation plans only carry K-side geometry; the `N`-side fields
of `TilePlan` (`logical_out_dim`, `row_tile_num`, `data_num`,
`n_pad`) are set to `0` on activation plans and consumers must
not read them.  No magic `n = 0` / `col_num = 1` placeholder
trick — the two paths are first-class.

## `XbarMapper` interface

Every runtime method takes the **same full xbar capability set**
as keyword-only args (no ``**kwargs``):

```python
class XbarMapper(ABC):
    def x_value_range(self, *, x_range, col_num, row_num,
                       w_digit_count, w_digit_radix, w_digit_range) -> tuple[int, int]: ...
    def w_value_range(self, *, x_range, col_num, row_num,
                       w_digit_count, w_digit_radix, w_digit_range) -> tuple[int, int]: ...
    def x_slice_radix(self, *, x_range, col_num, row_num,
                       w_digit_count, w_digit_radix, w_digit_range) -> int: ...
    def w_slice_radix(self, *, x_range, col_num, row_num,
                       w_digit_count, w_digit_radix, w_digit_range) -> int: ...
    def map_x(self, x: Tensor, *, x_range, col_num, row_num,
                  w_digit_count, w_digit_radix, w_digit_range) -> XMappingResult: ...
    def map_w(self, w: Tensor, *, x_range, col_num, row_num,
                  w_digit_count, w_digit_radix, w_digit_range) -> WMappingResult: ...
```

The macro never picks a subset.  Implementations forward only
what they use (a concrete mapper may free-form `del` the kwargs
it ignores).

Concrete impl: **`SimpleMapper(*, tiler, x_slicer, w_slicer)`**.

## Mapper → slicer translation

`SimpleMapper` is the translation layer that adapts the macro's
xbar capabilities to the slicer's unified 3-kwarg contract.  It
never reaches into a slicer's private signature; every slicer
call documents the three mandatory kwargs inline.

Activation path:

```python
self._x_slicer.slice(
    x,
    digit_count=1,
    digit_radix=x_hi - x_lo + 1,
    digit_range=(x_lo, x_hi),
)
```

Weight path:

```python
self._w_slicer.slice(
    w,
    digit_count=w_digit_count,
    digit_radix=w_digit_radix,
    digit_range=w_digit_range,
)
```

`mapper.x_value_range(...)` / `mapper.w_value_range(...)` forward
to the corresponding slicer methods using the same translation; no
caller-supplied `value_range` is ever fabricated.  A different
slicer drops in with no mapper-side changes — the mapper only sees
the abstract `Slicer` contract.

## Data flow inside `SimpleMapper`

### Weight path (`map_w`)

```
[Bw, N, K]
    -> w_slicer.slice(...)                                       # [Bw, N, K, Sw, D]
    -> tiler.tile_w(..., plan=tiler.make_w_plan(...))            # [Bw, Tr, data_num, Tc, row_num, Sw, D]
    -> permute + insert M=1, Sa=1                                 # [Bw, M=1, Tc, Tr, Sa=1, Sw, data_num, D, row_num]
```

### Activation path (`map_x`)

```
[Bx, M, K]
    -> x_slicer.slice(...)                                       # [Bx, M, K, Sa, digit_num=1]
    -> tiler.tile_x(..., plan=tiler.make_x_plan(...))            # [Bx, M, Tc, row_num, Sa, 1]
    -> squeeze(-1) + transpose + insert Tr=1, Sw=1
                                                                  # [Bx, M, Tc, Tr=1, Sa, Sw=1, row_num]
```

## Range terminology

* `xbar` exposes primitive grids: `x_range`, `w_digit_range`.
* Slicer / mapper / macro use ``value_range``:
  - `slicer.value_range(...)` — algorithm-side range covered by
    that slicer's strategy.
  - `mapper.x_value_range(...)` / `mapper.w_value_range(...)`.
  - `macro.x_value_range` / `macro.w_value_range` (properties
    that forward to the mapper after reading the xbar's
    capabilities).
* `transcoder.value_range()` — base envelope of one positional-
  radix digit string, the single source of truth slicers consume.

## `XbarMacro` composition

```python
XbarMacro(
    xbar=...,             # neurox.xbar.Xbar
    mapper=...,           # XbarMapper (single)
    col_accumulator=...,  # neurox.digital.Accumulator
    w_shift_adder=...,    # neurox.digital.ShiftAdder
    x_shift_adder=...,    # neurox.digital.ShiftAdder
    requantizer=...,      # neurox.digital.Requantizer
)
```

Construction is pure composition — no mapping-strategy kwargs on
the macro.  Inside `fabricate` / `matmul` the macro snapshots
xbar capabilities to local variables and passes each by **explicit
keyword argument** to every mapper call:

```python
x_range       = self.xbar.x_range
col_num       = int(self.xbar.col_num)
row_num       = int(self.xbar.row_num)
w_digit_count = int(self.xbar.w_digit_count)
w_digit_radix = int(self.xbar.w_digit_radix)
w_digit_range = self.xbar.w_digit_range

self.mapper.map_w(
    weight,
    x_range=x_range, col_num=col_num, row_num=row_num,
    w_digit_count=w_digit_count, w_digit_radix=w_digit_radix,
    w_digit_range=w_digit_range,
)
self.mapper.map_x(
    input,
    x_range=x_range, col_num=col_num, row_num=row_num,
    w_digit_count=w_digit_count, w_digit_radix=w_digit_radix,
    w_digit_range=w_digit_range,
)
```

Every call site documents its full argument list inline; the
macro has no helper dict to splat.  A different mapper drops in
without changing this method body.

## Naming conventions

* Variable names — `x_slice_num`, `w_slice_num`,
  `col_tile_num`, `row_tile_num`.  No `_size`.
* Shape-annotation shorthand — `Sa`, `Sw`, `Tc`, `Tr` —
  only in comments / docstrings, never as program variables.
* Shape annotation format: `# Shape: [old] -> [new]` or
  `# Shape: [new]`.

## Capability classification

The six xbar primitive capabilities split by **semantic purpose**:

| Capability         | Group        | Meaning                                        |
| ------------------ | ------------ | ---------------------------------------------- |
| `x_range`          | value-domain | primitive input grid                           |
| `w_digit_count`    | value-domain | digit slots per xbar-word                      |
| `w_digit_radix`    | value-domain | per-digit radix                                |
| `w_digit_range`    | value-domain | one digit's physical range                     |
| `col_num`          | geometry     | xbar tile output-side size (= `data_num`)      |
| `row_num`          | geometry     | xbar tile input-side size                      |

`col_num` / `row_num` size the array, not the value grid; they are
**not** value-domain capabilities.  Docs and comments that called
them "value-domain capability" were wrong and have been cleaned.

## `IdealMacro` range surface

`IdealMacro` is the macro-level zero-cost twin of `XbarMacro`.  To
keep the two macro families parallel at the operator boundary,
`IdealMacro` accepts the value ranges **directly** as constructor
arguments — no bit-width fiction:

```python
IdealMacro(
    x_value_range=(-127, 127),
    w_value_range=(-127, 127),
)
```

The tuple form mirrors what `XbarMacro.x_value_range` /
`XbarMacro.w_value_range` publish via their mapper; operators see
the same range API regardless of which macro family backs them.

## Refactor roadmap

Already landed:

* Full-capability kwarg set for every mapper method.
* Unified 3-kwarg slicer contract — no ``**kwargs`` variadics, no
  caller-supplied `value_range` input.
* `value_range` is the slicer's *output* semantics — derived from
  static strategy + the three runtime kwargs.
* `SimpleMapper` translates xbar caps → slicer kwargs; no slicer
  private signatures, no value_range placeholder fabrication.
* Explicit-arg call sites everywhere; no ``**dict`` splats.
* Separate `make_w_plan` / `make_x_plan` factories.
* `Transcoder.value_range()` consumed by slicers.
* `SimpleSlicer.value_range()` reachable-domain contract.
* `IdealMacro` accepts explicit value-range tuples.
* `col_num` / `row_num` reclassified as geometry capability.

Next steps land behind the same interfaces:

1. **Tensor-driven `w_shift_adder` reduction** — let the digital
   pipeline consume `slice_weights` (and `digit_weights`)
   directly rather than a scalar `slice_radix`.
2. **Per-layer heterogeneous strategy** — each layer can
   construct its own `SimpleMapper` with its own
   `slice_num` / `encoding` choices; no macro-side plumbing
   is needed.
