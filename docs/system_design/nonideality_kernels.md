# Non-ideality composition

A non-ideality combines a statistical law with a physical source and an event lifetime. The law describes the perturbation's distribution; the physical model determines its magnitude and dependence on operating conditions. The shared laws are specified in [non-ideality](../reference/primitive/nonideality.md).

## Source ownership

Each perturbation belongs to the component that models its physical cause. A fixed spread follows the design parameters, while a state-dependent spread follows the relevant operating condition. Combining sources preserves these distinct causes without applying one physical deviation at multiple owners.

## Lifetime and correlation

The same distribution can describe either static mismatch or dynamic fluctuations. Its lifetime follows the physical event that creates it, and its correlation follows which observations share that event. These choices obey [physical state](physical_state.md) independently of the selected statistical law.

## Run choices

A run selects which sources contribute to the modeled behavior. An inactive source contributes no perturbation; an active source follows its specified law and event lifetime. Selecting sources changes the modeled effects while preserving the hardware design described in [construction](construction.md).
