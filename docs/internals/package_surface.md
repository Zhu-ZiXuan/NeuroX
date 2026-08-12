# Package surface

The package surface is the public import boundary each `neurox/**/__init__.py` exposes, so implementation files and subdirectories can mirror the model hierarchy while the names callers import stay stable. This public-API boundary follows the standard Python idiom — a package's `__all__` / `__init__.py` surface (PEP 8). A name's **home directory** is the package whose own `.py` file defines it.

## Export law

- **Every `__init__.py` is a curated export face.** It may export its direct subpackages, and it may lift names from its own direct `.py` files that it intends to offer. Everything it offers is listed in `__all__`.
- **A face never forwards a subpackage's contents upward.** A name defined inside a subpackage is reachable only at its home directory: `neurox` exports `primitive`, while `Iadc` is importable only from `neurox.primitive.analog.current_adc`. A parent package therefore never degrades into a flat catalog of every descendant symbol.
- **Two exceptions are sanctioned.** `neurox/__init__.py` re-exports exactly the user-facing trio `Profiler`, `Reporter`, and `stamp_names` from the `neurox.common` face, plus its subpackages, and nothing else. `neurox/primitive/__init__.py` lifts `T_ROOM__K` individually from `physics.py`; the other `physics.py` names stay kernel-internal.

## Import layering

These rules bind every module inside `neurox/`, including `neurox/works/` and `neurox/tools/`. Standard-library and third-party imports are untouched by them.

- **Cross-module imports are absolute and end at the target's home directory.** `from neurox.architecture.unit.cim import CimUnit`. The path never ends at a `.py` file, and never at the top-level `neurox` package.
- **Same-directory imports are relative.** `from .base import Name` for a sibling file, `from .engine import Name` for a child subpackage's face. A relative path never pierces into a subpackage's files (`from .engine.base import Name`).
- **A descendant reaches an ancestor-package file directly.** `from neurox.architecture.unit.base import UnitBase` inside `neurox/architecture/unit/cim/base.py`, where the ancestor is a package on the importer's own path to the root. This is the one sanctioned file-terminated form: the directory door would re-enter a partially initialized ancestor `__init__` and cycle.

## Crossing the boundary

- **Implementation files are not the user API.** Examples, guides, and library users import from package surfaces, not from leaf `.py` files.
- **Tests may cross the boundary deliberately.** A test may import an implementation file when it needs a private seam; application and library users stay on the package surfaces.
- **A private seam is never documented as a user path.** A test importing `neurox.<pkg>.<impl_file>` does not make that path public, and it must not be presented as one.

## Packaging responsibility

A package surface is populated by that package's own `__init__.py` importing the implementation modules at its directory level and listing their public names in `__all__`. A family root package additionally imports each of its concrete implementation modules, so the family's full set of public types is reachable from the family surface.

## Registration completeness

Registration is an import side-effect: a concrete implementation registers itself when its module is first imported. Every package on the model tree explicitly imports its direct subpackages, so `import neurox` recursively loads the complete core and bundled-work implementation tree, including `neurox.works`. Each package lists those subpackages in `__all__`; callers and tools do not perform registration-only imports.
