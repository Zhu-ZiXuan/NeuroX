# Get started

Run an integer linear operator and inspect its profile on CPU. This example uses an ideal implementation to make the arithmetic easy to check; its configured area and leakage are zero and its modeled latency is zero.

## Install from a checkout

The supported Python and dependency ranges are declared in `pyproject.toml`. With a compatible Python environment activated, install the library:

```bash
python -m pip install .
```

For an isolated environment managed with uv, run these commands from the checkout root:

```bash
uv venv
uv pip install .
```

Activate that environment, or use its Python executable to run the code below.

## Program and execute an operator

Save this code as `quickstart.py` and run it with the installed environment's Python:

```python
import torch

from neurox import Profiler, Reporter, set_profile_leading_rank
from neurox.architecture.unit.linear import (
    IdealLinearUnit,
    IdealLinearUnitConfig,
    IdealLinearUnitPolicy,
)

unit = IdealLinearUnit(
    config=IdealLinearUnitConfig(
        x_value_range=(0, 3),
        w_value_range=(-1, 1),
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
    ),
    policy=IdealLinearUnitPolicy(),
    w_logical_shape=(2, 3),
)
weight = torch.tensor([[1, -1, 0], [0, 1, 1]], dtype=torch.int64)
unit.program(weight)

x = torch.tensor([[2, 2, 3]], dtype=torch.int64)
set_profile_leading_rank(unit, 1)  # Preserve the batch axis in observations.
profiler = Profiler(concat_dim=0)
profiler.collect_static_data(unit)
with profiler:
    output = unit.linear(x, quantization_mode=0)

torch.testing.assert_close(output, torch.tensor([[0, 5]], dtype=torch.int64))
report = Reporter(profiler.result)
print(output.tolist())
print(report.breakdown("area"))
print(report.breakdown("working_duration"))
```

The numerical output is `[[0, 5]]`. The root unit's name in the report is the empty string. Its area and working duration are zero under this ideal model; these values are not an estimate for a physical circuit. A missing energy contribution remains unavailable rather than demonstrating zero physical energy.

## Use a hardware model

Continue with the [operator workflow](../guides/algorithm_engineer/workflow.md) to select a CIM implementation, bind configuration and policy, establish physical state, and interpret its observations. The [unit API](../api/units.md) documents shapes, encodings, and lifecycle requirements. [Validation campaigns](../validation/campaigns.md) provide configured experiments for bundled circuit designs.
