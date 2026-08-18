# Writing system design pages

## Scope

A system design page states a software contract that spans components: what several modules must jointly honor for the program to stay correct, written once at the boundary instead of in every participant. It is engineer voice about the program, never the science the program implements and never a companion page for one module.

## Admission

The placement criterion — which carrier owns a given piece of knowledge, and what a page must satisfy to be admitted here — is stated once in [organizing_principles](../conventions/organizing_principles.md). Apply it before writing; a page exists only for knowledge that criterion sends here. The operative test at authoring time: a page earns its place when changing one end obliges the maintainer to understand the other and no single symbol can hold the rule.

## Page organization

A page answers a maintainer question — how construction propagates, where the compile boundary sits, who owns physical state — and organizes freely around that answer. Length follows the contract, and the heading structure follows the argument.

- No template and no prescribed section set: a page carries exactly the sections its subject needs.
- No traceability footer and no `N/A` placeholder: an absent topic is absent, not declared empty.
- No coverage expectation: a component with no cross-component contract gets no page, and no page is created merely because a module, a class, or a directory exists.
- No member catalogs: the code and the API index own the inventory of who implements a surface.
- Links are ordinary content, placed where a sentence depends on the target, under the same link rules as any other page — not an index block and not a fixed related-document list.

## What does not belong

- Single-class detail — a construction argument, one method's contract, one implementation's failure mode. It belongs to that class's docstring, which sits beside the code that enforces it.
- Control-flow narration. State the reason and the invariant the reader cannot derive; the code already shows the steps.
- Lessons learned, traps already fixed, and any other history: every carrier describes the present state only.
- Proposals that are not implemented, and alternatives that were rejected.
- One-time programming decisions — a choice settled by the code that implements it, whose change would be an explicit redesign that review catches on its own.
