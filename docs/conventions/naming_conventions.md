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

A code module file carries a leading underscore when it is private to its package — when it is the code file of no self-identifying type and nothing outside its own package subtree depends on it. A package subtree is a module's directory together with that directory's subpackages; a sibling package or an ancestor's other branch lies outside it. A module that is the code file of a self-identifying type, or whose helpers are used by code outside its subtree, is named bare. Tests and scripts may reach a private module directly; such use does not make it public. A leading underscore worn only because the name would otherwise begin with a digit is lexical, not a privacy marker.
