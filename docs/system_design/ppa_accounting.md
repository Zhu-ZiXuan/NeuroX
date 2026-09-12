# PPA accounting

PPA accounting associates area and leakage with physical hardware, and dynamic energy with the events that use it. The execution schedule determines how often hardware is reused; ownership determines how many hardware instances exist.

## Hardware ownership

Each component accounts for its own area and leakage once. A composite combines the contributions of its owned components with its local costs. A shared source or driver retains one owner even when several components use it.

Physical replication increases hardware cost. Temporal reuse increases the number of access events while retaining the same hardware. Grouping operations into one simulation call does not change either count.

## Access energy

Each energy contribution belongs to the physical activity it models. In an array access, establishing a held boundary is counted once, while excursions through successive operating phases contribute according to the phase schedule. Boundary drivers and the array account for their respective circuit costs.

Electrical energy is evaluated from the operating behavior of an access. Numerical iterations used to find that behavior do not represent additional physical accesses and do not multiply its energy. Chunking preserves the same set of contributions while changing how the calculation is partitioned.

## Measurement and aggregation

A measurement retains the distinction between independent caller operations and the internal work that each operation causes. Aggregation combines internal contributions into the corresponding operation total, including physical multiplicity and temporal reuse exactly once.

Enabling measurement preserves the execution's electrical behavior and physical sampling. Reports describe the measured hardware and events; the act of recording them introduces no modeled circuit activity.
