# Ye2023 JSSC validation

This campaign compares the WH-2T1R macro with the measured power breakdown in Fig. 19. It models the published configuration without redundant sub-array mapping. Static energy and average power use the validation-only 85 ns measurement cycle; this period is not a runtime macro interface.

The 50% input-sparsity point is the calibration anchor and the only hard gate. The 87.5% point is an informational extrapolation because the two published breakdowns cannot be reproduced by one activity-independent accounting boundary.

Run the record workload with:

```bash
make validate_ye2023jssc
```

The command writes `validation.log` and `power_breakdown.svg` under `log/validation/ye2023jssc/`.
