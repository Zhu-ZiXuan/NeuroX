# Application API

The root `neurox` package exposes construction, lifecycle, profiling, and reporting entry points. [Compute units](units.md) provide numerical operators. The contracts below are generated from source; [PPA accounting](../system_design/ppa_accounting.md) explains their physical interpretation.

## File construction

::: neurox.api.factory

## Assembled-model operations

::: neurox.api.function

## Profiling

::: neurox.api.profiler
    options:
      members: [Profiler, ProfileItem]
      inherited_members: [__enter__, __exit__]

## Reporting

::: neurox.api.reporter
