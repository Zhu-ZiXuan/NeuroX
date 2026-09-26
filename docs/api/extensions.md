# Extension interfaces

These interfaces support new models, family implementations, and reusable numerical tools. Read the owning class and method contracts before implementing a hook. [Construction](../system_design/construction.md), [physical state](../system_design/physical_state.md), and [PPA accounting](../system_design/ppa_accounting.md) explain their interaction across components.

## Modules and companion data

::: neurox.common.module
    options:
      members: [ModuleBase, ProfileModule, NonProfileModule, ConfigBase, PolicyBase, SnapBase, DcopBase]
      filters: ["!^__"]

## Construction and validation

::: neurox.common.base_only_mixin

::: neurox.common.registry_mixin
    options:
      filters: ["!^__"]

::: neurox.common.validate_mixin
    options:
      filters: ["!^__"]

::: neurox.common.serialize_mixin

::: neurox.common.serialize

## Tensor containers and recording

::: neurox.common.dataclass_mixin

::: neurox.common.recorder.RecorderBase
    options:
      members: [current, active, result, submission, _merge_records, _export_tensor, _submit_record]

## Encoding and mapping

::: neurox.encoding

::: neurox.architecture.mapping.slicer.base

::: neurox.architecture.mapping.slicer.direct

::: neurox.architecture.mapping.slicer.simple

::: neurox.architecture.mapping.tiler.base

::: neurox.architecture.mapping.tiler.simple

::: neurox.architecture.mapping.input_phase_splitter

::: neurox.architecture.mapping.merge

::: neurox.architecture.unit.cim

## Numerical execution

::: neurox.common.torch_compat

::: neurox.execution.chunking

::: neurox.execution.loop

::: neurox.execution.solving

## Rounding

::: neurox.common.quantization
