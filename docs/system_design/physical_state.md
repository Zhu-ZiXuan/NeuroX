# Physical state

Physical state belongs to the hardware component whose behavior it describes. Its lifetime determines which observations share a realization and which events introduce fresh variation.

## Lifetimes

- **Design values** describe the nominal hardware and remain fixed across realizations of that design.
- **Fabricated state** describes one manufactured instance, including static mismatch. It remains fixed until a new realization is selected.
- **Programmed state** describes the value stored in programmable hardware, including programming effects. It remains fixed until the next programming event.
- **Sampled state** describes a particular access, including its dynamic fluctuations. Its lifetime follows the physical event being modeled.

A run establishes a hardware realization, programs the components that require stored values, and executes accesses against that state. Repeating fabrication selects a realization from the same nominal design; repeating programming replaces the stored value according to the programming model. Each owner determines how a logical programming operation maps onto its children.

## Temperature and explicit lifecycle events

Temperature belongs to the run environment. Setting it establishes the temperature that subsequent computations read, without sampling randomness. Fabrication and programming each use the environment established before that explicit event; their materialized results remain fixed until the corresponding event is triggered again. Each access computes its runtime coefficients from the current temperature and uses them with those retained results.

The caller controls when to fabricate and supplies the input for each programming event. Temperature updates neither trigger these events nor retain programming inputs for replay. Snapshots and operating points already produced belong to their original access.

## Sampling and correlation

Fabrication samples over physical instances. Access sampling additionally distinguishes events in time. Reusing one hardware instance preserves its fabricated mismatch while allowing fresh dynamic fluctuations on subsequent accesses.

The component scheduling an event determines which participants share it and which observations are independent. A held boundary may be shared across several phases of an access; that sharing is part of the circuit schedule. Consumers use the supplied realization throughout their evaluation.

Numerical batching and chunking partition existing work. They create neither hardware instances nor physical accesses, and they preserve sampled values and their correlation. Access sampling therefore precedes partitioning the numerical solve. Iterations toward one equilibrium use the same sampled physical state.

## Shared sources and consumers

A shared reference has one physical identity and one fabricated realization. Every consumer of that reference sees the same source deviation. A deviation specific to one reading circuit belongs to that circuit, while its per-access fluctuations follow its own access events.

This separation keeps source variation, consumer mismatch, and dynamic noise attributable to their physical owners. A component consumes another component's exposed state without maintaining a second physical realization of it.
