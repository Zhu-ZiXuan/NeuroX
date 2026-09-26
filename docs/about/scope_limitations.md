# Scope and limitations

NeuroX combines physical component models with integer operators and hardware accounting. Each model's assumptions and validity conditions in the [scientific reference](../reference/README.md) govern its use; a composite inherits the restrictions of its components.

## Numerical and physical behavior

Ideal operators provide integer arithmetic controls. CIM execution additionally depends on configured value ranges, mapping, converter behavior, and the selected non-idealities. Agreement with an ideal operator checks arithmetic under those assumptions; it does not establish agreement with silicon.

Device and circuit results depend on model equations and parameter provenance. Solver convergence establishes a solution to the configured equations within its tolerances, rather than validating the physical assumptions. Consult the relevant device, array, and solver references for the available assumptions and characterized conditions.

## Hardware statistics

Area and leakage describe configured physical hardware. Dynamic energy describes modeled events; adopted or calibrated circuit costs retain their stated provenance. A missing quantity is unknown, while zero is a modeled value. Ideal configurations with zero costs provide no physical PPA estimate.

Operation duration describes a unit's modeled work. [PPA accounting](../system_design/ppa_accounting.md) explains powered windows, shared hardware, and aggregation. Whole-model schedules, overlap, powered idle intervals, and hardware outside the configured tree require application-level modeling.

## Evidence

The [validation campaigns](../validation/campaigns.md) identify paper-specific configurations, targets, assumptions, and checks. A calibrated quantity is not independent evidence for the target used to fit it. Assess a result together with its configuration, policy, dependency environment, device, and validation outcome.

Some model pages still lack quantitative validity bounds, citations, or independent physical evidence. Specific unresolved items are retained on those pages; an equation or passing numerical test alone does not resolve them.
