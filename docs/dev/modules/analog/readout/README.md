# ReadOut Family

The `ReadOut` family owns the voltage-domain chain between an array's per-column boundary current / voltage and the final ADC codes.

Rules:

- grouped-lattice semantics belong at the readout boundary
- leaf circuits own electrical behaviour
- the readout container only performs orchestration, shape movement, and aggregation
- concrete readout configs carry concrete member configs for every owned block

Current concrete implementation:

- `OffsetSwitchCapMuxAdcReadOut`
