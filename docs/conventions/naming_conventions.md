# Naming and titles

Names identify concepts consistently across code and documentation.

Identifiers and filenames use concise English words and established ASCII domain abbreviations. Reuse the code name assigned to a scientific quantity by [notation conventions](notation_conventions.md); do not translate one concept into multiple synonymous names.

One identity names an object across code and documentation. A name read in isolation is self-identifying and carries the full identity, family included; a name read inside its package or directory is package-relative and drops the qualifier the enclosing name already supplies. A class name and a document title are self-identifying; a code module file is package-relative. With no enclosing family the two coincide.

Topic documentation files use snake_case. Standard repository files retain their recognized names, including `README.md`, `CONTRIBUTING.md`, `LICENSE`, and `CITATION.cff`.

| | Class | Code file | Title |
| --- | --- | --- | --- |
| **Standalone module** | `ModuleName` | `module_name.py` | Module name |
| **Family member** | `SchemeFamily` | `family/scheme.py` | Scheme family |
| **Family root base** | `Family` | `family/base.py` | Family base |

A family that occupies its own package places its root base in `base.py`, and the base class is the family's self-identifying name, kept unambiguous when the bare family word is too generic to stand alone. The shared science of a family is `family.md` in Reference, titled `<Family> family`, mirroring the package rather than a file since shared science spans files.

## Parallel bases qualified by topology

When one family splits into sibling abstract bases that are parallel — neither wraps nor specializes the other, and each stands alone against the same interface — an intrinsic distinguishing property qualifies each base name, so the name identifies its variant read in isolation. The qualifier names a structural or domain property fixed by the topology, such as the input's single-ended versus differential form, never a parameter value a config could carry. Parallel sibling bases share the family word and differ only by that qualifier prefix.

## Domain qualifiers

A domain qualifier is determined by the scope in which a name must remain unambiguous. Inside a module devoted to one domain, omit a qualifier that merely repeats that domain. In an outer scope that handles several domains, retain the qualifier that distinguishes otherwise ambiguous values. Moving a name across that boundary may therefore add or remove the qualifier without changing the concept it names.

## Python symbol prefixes

A single leading underscore marks a symbol outside the application API. A base may explicitly document such a method as a subclass hook; its extension contract then belongs in the owning docstring. Incidental use by an internal collaborator or test does not make a helper an extension interface. A bare name marks a supported interaction surface whose audience follows the owning User API or Extension SPI contract.

A double leading underscore is reserved for the rare case where a broadly inherited base deliberately uses Python name mangling to prevent an accidental subclass collision. It is not a stronger privacy marker than a single leading underscore.

A code module file carries a leading underscore when it is private to its package — when it is the code file of no self-identifying type and nothing outside its own package subtree depends on it. A package subtree is a module's directory together with that directory's subpackages; a sibling package or an ancestor's other branch lies outside it. A module that is the code file of a self-identifying type, or whose helpers are used by code outside its subtree, is named bare. Tests and scripts may reach a private module directly; such use does not make it public. A leading underscore always carries this privacy meaning; a source-file name that would otherwise begin with a digit takes a meaningful alphabetic prefix instead. (Machine-checked.)

## Tool configuration classes

A tool module carries its own config class, named by one of two patterns:

- **`Calibrate<Package><Module>Config`** — for a tool module named after the modelled family it calibrates. The qualifier taken from the tool package keeps the class distinct from the calibrated family's own `<Module>Config`, which the same tool loads alongside it.
- **`<Module>ToolConfig`** — for a tool module named after a procedure stage of its own, one of several stages composing a multi-step tool. The stage name is self-identifying already, and `Tool` marks the class as that tool's config rather than a modelled object's.

## Generic type parameters

A generic type parameter is named `[<Owner>]<Role>T`: an optional ownership qualifier, a role stem, and the trailing `T`. The role stem is canonical and fixed per role — `ConfigT`, `PolicyT`, `SnapT`, `DcopT`, `RecordT`, `MeasureT`, `ModuleT`, `NodeT`, and a bare `T` for a utility generic over a type it never inspects.

The qualifier answers whose companion type the parameter stands for. An unqualified name is the declarer's own companion type, or a role that is ownership-agnostic to begin with. A qualified name is another module's companion type — a composed child or a collaborator — and the qualifier is that owner's name: `<Child>SnapT` is the child's snap, `<Collaborator>RecordT` is the collaborator's record.

When several type parameters have the same role, qualify them by owner; at most one remains unqualified.
