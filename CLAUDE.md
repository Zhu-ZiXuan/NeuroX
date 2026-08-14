Follow the best practice of key libraries.
Use new features and new APIs.
No compatibility to older versions.
Assume `torch.compile` is applied to caller of functions for performance optimization.
No need to clean all ruff or mypy error or warning, treat them as helper.
Never use `# noqa:` or `# type:` comments to skip errors or warnings.
Existing suppression comments are the user's hand-added rulings, made after manual review: never remove, rewrite, or relocate one, and never treat one as precedent for adding more.
Check for available CUDA devices (both memory and utility) before running each task on GPU.
Update the docs when finish a task.
Always use English in code and doc, be concise and precise.
Never add unnecessary newlines in docs.
Read `docs/README.md` for the documentation map; the scientific spec is in `docs/reference/`, implementation internals in `docs/internals/`, and coding/doc standards in `docs/contributing/`.

Key library version:
- python 3.12+
- pytorch 2.10+

Utility `make` targets:
- `make format`: Run `ruff` formatter with auto fix.
- `make lint`: Run `ruff` linter.
- `make check`: Run `mypy` static analysis.
