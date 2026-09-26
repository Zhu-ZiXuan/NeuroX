# Compute units

Use linear and convolution families for integer operators. Their docstrings define programming, input layout, profiling requirements, and implementation-specific constraints. The [scientific model](../reference/architecture/unit/family.md) defines the arithmetic; [CIM mapping](../reference/architecture/unit/cim.md) describes hardware decomposition.

## Shared interface

::: neurox.architecture.unit.base

## Linear family and implementations

::: neurox.architecture.unit.linear.base

::: neurox.architecture.unit.linear.cim
    options:
      inherited_members: [program, linear, rescale_factor, w_value_range, x_value_range, adc_bits]

::: neurox.architecture.unit.linear.ideal
    options:
      inherited_members: [linear]

## Convolution family and implementations

::: neurox.architecture.unit.conv2d.base

::: neurox.architecture.unit.conv2d.cim
    options:
      inherited_members: [program, conv2d, rescale_factor, w_value_range, x_value_range, adc_bits]

::: neurox.architecture.unit.conv2d.ideal
    options:
      inherited_members: [conv2d]
