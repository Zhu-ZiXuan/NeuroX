"""Literature-derived schemes packaged with the core library.

Each scheme is a self-contained composition of kernel primitives, laid out
by circuit layer (for example :mod:`neurox.works.macro.cim`). Importing
:mod:`neurox.works` registers every scheme class with its family registry.
"""

from neurox.works import macro

__all__ = ["macro"]
