# Package surface

The package surface is the public import boundary each `neurox/*/__init__.py` exposes, so implementation files and subdirectories can mirror the model hierarchy while the names callers import stay stable. This public-API boundary follows the standard Python idiom — a package's `__all__` / `__init__.py` surface (PEP 8).

## Import layering

- **Cross-directory imports name a package, not a file module.** A symbol is imported from the package that owns the file where it is defined (`from neurox.<pkg> import Name`), never by reaching across a directory boundary into the defining `.py` module. Import paths stay predictable, and parent packages do not degrade into flat catalogs of every descendant symbol.
- **A family subpackage self-exports its surface; the parent does not re-expand it.** A subpackage's `__init__.py` owns the public surface at its own directory level and exports the public symbols defined there. The parent package imports that subpackage's surface as a unit and does not re-list the subpackage's descendants by reaching past it. Re-exporting from the current directory is an ordinary API decision; re-exporting a descendant from a parent is an exception that must be justified.
- **A parent re-exports a descendant only for a documented reason.** Two reasons are sanctioned. (a) The natural technical name is not a valid Python identifier — for example it would start with a digit — so the on-disk package takes the smallest underscore-prefixed spelling that makes it valid; that leading underscore also marks the package as internal to `ruff` and `mypy`, so the parent re-exports the package's limited surface under a stable, tool-clean name. (b) A subdirectory is only an internal grouping that sits behind a deliberately flat public surface. Convenience alone is never a reason: a parent re-export widens the public API and creates migration burden, so it is used only for one of these documented cases.
- **The top-level package presents a curated flat re-export.** The outermost `neurox/__init__.py` gathers a curated, flat public surface from the subpackages below it, so the most common entry points are importable directly from the top-level package — an instance of reason (b). This describes the current arrangement only.

## Crossing the boundary

- **Implementation files are not the user API.** Examples, guides, and core callers import from package surfaces, not from leaf `.py` files.
- **Tests and tools may cross the boundary deliberately.** Tests and developer tooling may import implementation files directly when they need a private seam; application and library users stay on the package surfaces.
- **A private seam is never documented as a user path.** A test importing `neurox.<pkg>.<impl_file>` does not make that path public, and it must not be presented as one.

## Packaging responsibility

A package surface is populated by that package's own `__init__.py` importing the implementation modules at its directory level and listing their public names in `__all__`. A family root package additionally imports each of its concrete implementation modules, so the family's full set of public types is reachable from the family surface.

## Registration completeness

Registration is an import side-effect: a concrete implementation registers itself when its module is first imported. Because a family root package imports its concrete modules, importing that package completes registration of the whole family; the registration and lookup mechanism is specified in [registry](common/mixin/registry.md). A family package therefore does not need to be re-exported by its parent for its registrations to complete: the parent imports the family package as a unit, and need not re-expand it into its descendants. No automated check verifies that a package's `__init__.py` imports keep its public surface — and therefore its registrations — complete.
