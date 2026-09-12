# Ye2023 calibration tools

The parallel BL/SL solver owns its stopping constants. Its ordinary path rejects a capped unconverged solve, while its explicitly traced path may return the failure for diagnosis. There are no user-supplied solver tolerances or tolerance-sweep tools, and calibration does not inspect solver traces to decide convergence.

No cell or ADC calibration script is needed. The divider table follows from the reported RRAM resistances and the geometric-mean T1 bias; the unit-width T2 leakage and signal current come from Fig. 16. The 7 uA RS-CSA reference comes from Fig. 11 and gives a rescale factor of 14 MAC units per full-resolution code.
