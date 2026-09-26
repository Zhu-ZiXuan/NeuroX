# Construction

A simulation separates the hardware design from the choices made for a run. Construction combines those choices into a system whose components have explicit ownership and compatible responsibilities.

## Design and run choices

Configuration describes physical, design, and specification values reused across runs. Policy selects how a run models that design, including which non-idealities are active. A source's physical magnitude belongs to the design; the decision to apply it belongs to the run.

Both descriptions follow the system's composition. Each component receives the design and run choices relevant to its own responsibility. Alternative realizations of a component must satisfy the same role within the enclosing system.

Construction initializes each component at the shared default temperature. The caller sets the run temperature on the assembled system before the lifecycle events it should affect, under the [temperature and explicit lifecycle contract](physical_state.md#temperature-and-explicit-lifecycle-events).

## Composition and ownership

A composite determines which components it contains and how their capacities and physical multiplicities fit together. Each child interprets its own design parameters and exposes the behavior its parent needs. Derived quantities belong to the component that defines their meaning.

Ownership identifies a physical component once. Several components may use one shared source or peripheral without creating additional hardware instances. This distinction carries through the [physical-state lifecycle](physical_state.md) and [PPA accounting](ppa_accounting.md).

The user-facing representation of design and run choices is specified in [configuration](../api/configuration.md). A campaign also retains the provenance of its chosen values under [campaigns](../validation/campaigns.md).
