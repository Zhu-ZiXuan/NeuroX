# Transcoder Family

The transcoder is the **pure-algorithm** layer that converts an integer tensor into a fixed-length signed-digit list and back. It is shape-agnostic, hardware-agnostic, and exposes no notion of xbar tiling.

## Layout

```
neurox/mapper/transcoder/
├── base.py        # Transcoder ABC + Encoding alias + registry-backed create() factory
├── true_form.py   # TrueFormTranscoder, registered under "true_form"
├── complement.py  # ComplementTranscoder, registered under "complement"
└── canonical.py   # CanonicalTranscoder, registered under "canonical"
```

Each concrete subclass overrides the `encode()` method and the `value_range` `@property`; the ABC supplies the shared `decode()` (positional weighted sum) and the `radix` / `digit_num` properties.

## Encoding policies

| Encoding | Digit policy | Inclusive `value_range` |
|---|---|---|
| `true_form` | Sign-magnitude: digits share the input sign. | `[-(r^D - 1), r^D - 1]` |
| `complement` | Low `D-1` digits unsigned; MSB folds into `{-⌊r/2⌋, …, ⌈r/2⌉-1}`. | `[-⌊r/2⌋·r^(D-1), ⌈r/2⌉·r^(D-1) - 1]` |
| `canonical` | Non-adjacent-form-style: at most every other position non-zero. | `[-M, M]` with `M = Σ_j (r-1)·r^(D-1-2j)`, `j ∈ [0, ⌈D/2⌉)` |

`value_range` is the inclusive integer band each encoding can **losslessly** round-trip. Inputs outside that band silently wrap or truncate; callers must respect it.

## Discriminator-driven factory

`Transcoder` parametrises `RegistryMixin[Encoding, "Transcoder"]`. Each concrete subclass self-registers via a class decorator:

```python
@Transcoder.register_key("true_form")
class TrueFormTranscoder(Transcoder): ...
```

The ABC exposes a thin factory wrapper that looks the impl up by encoding and instantiates it:

```python
Transcoder.create(encoding, radix=r, digit_num=D)
```

Registration is driven by import side-effect; importing `neurox.mapper.transcoder` (which the subpackage `__init__.py` does) loads every concrete subclass and populates the registry.

Subpackage rule: import the subclass / factory via `from neurox.mapper.transcoder import …` — `neurox.mapper` itself never re-exports these.

## What the transcoder is not responsible for

- Tile-level layout, padding, or geometry.
- Per-value digit-count / digit-radix decisions — these arrive as arguments.
- Reference-column placement.
- Analog-domain digit weighting.

This is the smallest possible abstraction: take an integer plus a digit schema, produce the corresponding signed-digit list (and vice versa).

See also:

- `docs/modules/mapper/xbar/slicer/README.md`
- `docs/dev/architecture/mapping.md`
