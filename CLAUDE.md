Follow the best practice of key library.
Use new feature and new API.
Never provide compatibility to older versions.
Assume `torch.compile` is applied to caller of functions for performance optimization.
Write Google style docstring for core neurox simulator code, be concise and accurate.
No need to clean all ruff or mypy error or warning, treat them as helper.
Check for available cuda devices (both memory and utility) before running every task.
Update the docs when finish a task.
Always use English in code and doc, be concise and precise, no unnecessary newline.
Never link to other files, explain everything in comments or docstring.

Physical units:
- for tensors: voltage [V], current [uA], conductance [uS], resistance [MOhm], time [ns], capacitance [fF], energy [fJ], power [uW], length [um].
- for configs: follow industrial standards, or use tensor units when no standard.
- all unit conversion must be done during config extraction in init procedure.
- only do unit conversion in python flow.
- never do unit conversion during tensor computation.
- all tensor units must be correct at construction.

Key library version:
- python 3.11+
- pytorch 2.9+

Utility `make` targets:
- `make format`: Run `ruff` formatter with auto fix.
- `make lint`: Run `ruff` linter.
- `make check`: Run `mypy` static analysis.
