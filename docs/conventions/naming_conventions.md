# Naming and titles

The single home of the naming law — whether a name carries its family qualifier or drops it, for a class name, a code or doc file, and a document title.

One identity names an object across code and documentation. A name read in isolation is self-identifying and carries the full identity, family included; a name read inside its package or directory is package-relative and drops the qualifier the enclosing name already supplies. A class name and a document title are self-identifying; a code module file and its mirroring document file are package-relative. With no enclosing family the two coincide.

Every documentation file is named in snake_case, and a Reference or Internals module document's filename matches its code module's filename exactly.

| | Class | Code and mirrored doc file | Title |
|---|---|---|---|
| **Standalone module** | `ModuleName` | `module_name.py` / `.md` | Module name |
| **Family member** | `SchemeFamily` | `family/scheme.py` / `.md` | Scheme family |
| **Family root base** | `Family` | `family/base.py` / `base.md` | Family base |

A family that occupies its own package places its root base in `base.py`, and the base class is the family's self-identifying name, kept unambiguous when the bare family word is too generic to stand alone. The shared science of a family is `family.md` in Reference, titled `<Family> family`, mirroring the package rather than a file since shared science spans files. Reference carries `family.md` and Internals `base.md` — one science, one software, related through the footer rather than a shared title. A leaf and its module document, the two faces of one object, carry a verbatim-identical title.

## Parallel bases qualified by topology

When one family splits into sibling abstract bases that are parallel — neither wraps nor specializes the other, and each stands alone against the same interface — an intrinsic distinguishing property qualifies each base name, so the name identifies its variant read in isolation. The qualifier names a structural or domain property fixed by the topology, such as the input's single-ended versus differential form, never a parameter value a config could carry. Parallel sibling bases share the family word and differ only by that qualifier prefix.

A code module file carries a leading underscore when it is private to its package — when it is the code file of no self-identifying type and nothing outside its own package subtree depends on it. A package subtree is a module's directory together with that directory's subpackages; a sibling package or an ancestor's other branch lies outside it. A module that is the code file of a self-identifying type, or whose helpers are used by code outside its subtree, is named bare. Tests and scripts may reach a private module directly; such use does not make it public. A leading underscore worn only because the name would otherwise begin with a digit is lexical, not a privacy marker.

## Generic type parameters

A generic type parameter is named `[<Owner>]<Role>T`: an optional ownership qualifier, a role stem, and the trailing `T`. The role stem is canonical and fixed per role — `ConfigT`, `PolicyT`, `SnapT`, `DcopT`, `RecordT`, `MeasureT`, `ModuleT`, `NodeT`, and a bare `T` for a utility generic over a type it never inspects.

The qualifier answers whose companion type the parameter stands for. An unqualified name is the declarer's own companion type, or a role that is ownership-agnostic to begin with. A qualified name is another module's companion type — a composed child or a collaborator — and the qualifier is that owner's name: `<Child>SnapT` is the child's snap, `<Collaborator>RecordT` is the collaborator's record.

Same-role parameters appearing together in one parameter list therefore differ in ownership, the qualifier is what tells them apart, and at most one of them is unqualified. If a parameter for the declarer's own companion type ever seems to need a qualifier, the class holds two selves; the design changes, not the name.
