# Offline tool API

`neurox.tools` provides offline task capabilities over the model API. Shared runtime modules serve both calibration and validation; each concrete calibration package also supplies a `python -m` command. Validation campaigns compose the reusable measurement and runtime interfaces in their own scripts.

## Runtime

::: neurox.tools.cli

::: neurox.tools.config

::: neurox.tools.module

::: neurox.tools.logging

::: neurox.tools.output

::: neurox.tools.run

## Calibration

::: neurox.tools.calibration.cli

::: neurox.tools.calibration.run

::: neurox.tools.calibration.output

::: neurox.tools.calibration.adc.config

::: neurox.tools.calibration.adc.run

::: neurox.tools.calibration.adc.data

::: neurox.tools.calibration.cim_macro.sampling

::: neurox.tools.calibration.cell.x1t1r

::: neurox.tools.calibration.cim_macro.construction

::: neurox.tools.calibration.cim_macro.math

::: neurox.tools.calibration.cim_macro.rescale_fit

## Validation

::: neurox.tools.validation.cim_macro.cli

::: neurox.tools.validation.cim_macro.construction

::: neurox.tools.validation.cim_macro.ppa
