# Parameter provenance

Reference parameter tables distinguish a quantity's source from its physical constraint.

## Source

| Source | Meaning |
| --- | --- |
| Constant | Universal physical constant; identify the adopted reference and whether its value is exact. |
| Measured | Physical measurement, with conditions and uncertainty where available. |
| Process | Nominal foundry, PDK, or datasheet value. |
| Extracted | Value obtained from layout or parasitic extraction. |
| Design | Value chosen within the model's permitted design space. |
| Calibrated | Value fitted against an identified physical or numerical reference. |

For a derived parameter, identify its upstream source and derivation. Distinguish physical-data fitting from numerical solver controls. A source category describes provenance; it does not establish accuracy at every operating point.

The [campaign tags](../validation/campaigns.md#provenance-tags) identify the provenance of concrete values in experiment files. Reference tables describe model parameters rather than a particular configured experiment.

## Constraint

State the physical domain accepted by the model. Use `—` when no additional physical bound is imposed; retain a specific unresolved bound when evidence is missing. Exception types and implementation validation belong to the owning interface.

Activations, programmed weights, temperature, and operating points are runtime inputs rather than entries in a design-parameter table.
