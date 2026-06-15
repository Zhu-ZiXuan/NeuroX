Follow the best practice of key library.
Use new feature and new API.
Never provide compatibility to older versions.
Assume `torch.compile` is applied to caller of functions for performance optimization.
No need to clean all ruff or mypy error or warning, treat them as helper.
Check for available cuda devices (both memory and utility) before running every task.
Update the docs when finish a task.
Always use English in code and doc, be concise and precise.
Never add unnecessary newlines in docs.
Never change `git` status in any condition.
Read `docs/dev/architecture.md` for detail requirements for this project.

Key library version:
- python 3.11+
- pytorch 2.9+

Utility `make` targets:
- `make format`: Run `ruff` formatter with auto fix.
- `make lint`: Run `ruff` linter.
- `make check`: Run `mypy` static analysis.
