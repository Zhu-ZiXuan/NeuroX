# CircuitBase and CircuitConfig

`neurox/common/circuit.py` defines the typed base every electrical circuit
inherits and the static-PPA-carrying config base every circuit's config
inherits.

## Why a base, not a mixin

`CircuitBase` makes the canonical mixin composition explicit and
re-usable. It **inherits** `FabricateMixin + ProfileMixin + nn.Module`
in that order, and adds `Generic[ConfigT]` so the typed config narrows
in subclasses. The mixins remain individually meaningful; `CircuitBase`
is just the standard composition every electrical circuit needs.

Splitting static PPA into its own mixin separate from `ProfileMixin`
would impose a load-bearing implicit contract (the profile-event side
would read PPA fields its body does not declare) and force each leaf
to compose the mixins by hand. The single base avoids both costs.

## Class shape

```python
@dataclass(frozen=True)
class CircuitConfig(ValidateMixin):
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    def validate_ppa(self) -> None: ...

ConfigT = TypeVar("ConfigT", bound=CircuitConfig)

class CircuitBase(FabricateMixin, ProfileMixin, nn.Module, Generic[ConfigT]):
    config: ConfigT
    _inst_shape: tuple[int, ...]
    def __init__(self, *, config, name, inst_shape) -> None: ...
    # 2 base properties: area_per_inst__um2 / leakage_per_inst__uW
    # 3 derived properties: inst_count / inst_area__um2 / inst_leakage__uW
    # 1 forwarder property: inst_shape (read-only over _inst_shape)
```

Per-op latency is **not** a base contract: leaves with a dynamic
profile model compute their own ``latency__ns`` tensor in the primary
method and emit it via ``_log_latency(latency)`` (paired with
``_log_dynamic_energy(dynamic_energy__fJ)`` when the leaf also emits
energy) — fixed-latency leaves declare ``latency_per_op__ns`` on their
own config and read it back as ``self.config.latency_per_op__ns``;
parametric leaves (e.g. ``McsSarAdc``) derive the value from runtime
parameters (active bit width × clock period, etc.). Leaves without a
dynamic model (`Driver`, `OpAmpTIA`) emit nothing — their physical
contribution folds into the owning circuit's energy / latency
tensors.

The base `__init__` explicitly drives `nn.Module.__init__(self)` and
`ProfileMixin.__init__(self, name)`, then binds `config` and
`_inst_shape`. Mixins without `__init__` (`FabricateMixin`) pass through
naturally.

## How leaf circuits use it

```python
class Driver(CircuitBase[DriverConfig]):
    nominal_drive_value: Tensor

    def __init__(self, *, config: DriverConfig, policy: DriverPolicy,
                 name: str, inst_shape: tuple[int, ...],
                 dtype: torch.dtype, T__K: float) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K
        self.register_buffer("nominal_drive_value", ..., persistent=False)
```

Three things happen automatically by composition:

1. `super().__init__` initialises `nn.Module` + `ProfileMixin`,
   binds the typed `config` and `_inst_shape`.
2. `self.area_per_inst__um2 / leakage_per_inst__uW` are inherited;
   they read from `self.config`. Per-op latency is *not* on the base —
   leaves that emit dynamic events read
   `self.config.latency_per_op__ns` (fixed) or derive it from runtime
   parameters at `_log_latency` emit time; non-emitting leaves
   (`Driver`, `OpAmpTIA`) carry no per-op latency at all.
3. `self.inst_count / inst_area__um2 / inst_leakage__uW` are derived on
   access (`math.prod(self._inst_shape)` × per-instance); no cache.

`mypy` knows `self.config: DriverConfig` from the `CircuitBase[DriverConfig]`
generic parameter; access `self.config.drive_value` types correctly. For
`nn.Module` subclasses, a class-level `config: DriverConfig` forward
declaration is sometimes needed when the parent's parameterised `Generic`
loses precision through `__getattr__` — the leaf carries that one-line
declaration alongside its buffer declarations.

## Per-circuit shape conventions

`CircuitBase` does not impose a single forward-tensor shape protocol; each
circuit family owns its own. The shared convention is:

- **Serial-op dims come before `_inst_shape`** in any forward input or
  output tensor.
- `_inst_shape` plus optional **per-circuit trailing dims** (e.g. `n_caps`
  on `SwitchCap`, `max_bits` on `ADC`, `digit_num` on `ReadOut`) describe
  in-instance parallel structure that does **not** scale latency.

A leaf that emits a latency event computes its own `serial_op_count`
from the forward tensor's shape minus its instance / parallel dims
(see each emitting leaf's `_log_latency` call site for the worked-out
arithmetic).

## What CircuitBase is NOT for

`CircuitBase` is for **electrical circuits** — modules that have a
per-instance static PPA cost and *may* emit per-op profile events
(some, like `Driver` / `OpAmpTIA`, carry static PPA only because their
dynamic contribution is folded into the owning circuit's energy /
latency tensors). It is NOT for:

- **Devices** (`RRAM` / `NMOS` / `Selector`): physical primitives without
  per-op latency. Their static cost rolls up into the owning circuit's
  `CircuitConfig`. They continue to inherit
  `FabricateMixin + nn.Module` directly. Their `*Config` does not inherit
  `CircuitConfig`.
- **Macros** (`XbarMacro` family): orchestration nodes that own zero
  silicon themselves and just dispatch into children. They do not
  inherit `CircuitBase` (they have no own config-backed PPA); they
  remain on `FabricateMixin + nn.Module + ProfileMixin + RegistryMixin`.
  Their PPA appears in the profile report through their constituent
  circuits, not through themselves.

## See also

- `docs/dev/architecture/profiler_and_ppa.md` — full PPA + profile event
  rules across the layer hierarchy.
- `docs/dev/modules/profiler/README.md` — `_log_dynamic_energy` /
  `_log_latency` side-channel and `NeuroxProfiler` aggregation.
- `docs/dev/modules/common/fabricate.md` — `FabricateMixin` cascade
  contract.
