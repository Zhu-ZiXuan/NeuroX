# `neurox/tools/logging.py`

## Current role

Shared logging setup for the offline tool CLIs under `neurox/tools/`. Centralises the `logging.basicConfig(...)` call so every tool emits in the same paste-ready format.

## Surface

One function:

```python
def config_tool_logging(level: int = logging.INFO) -> None: ...
```

- Installs a plain `"%(message)s"` format (no level / timestamp / logger-name prefix) so tool output is paste-ready into TOML / config files.
- Emits to stderr (the `logging.basicConfig` default).
- Idempotent — `basicConfig` is a no-op once the root logger has a handler, so repeat calls within the same process are safe.
- `level` defaults to `logging.INFO`; pass `logging.DEBUG` from a tool that wants verbose tracing.

## How tools use it

Every offline tool follows the same skeleton:

```python
import logging

from neurox.tools.logging import config_tool_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # ... argparse validation ...
    config_tool_logging()
    # ... tool body, logger.info(...) for every line of output ...
```

Tools never call `logging.basicConfig(...)` directly. New tools added under `neurox/tools/` must use this helper.

## Why this is not in `neurox/common/`

The helper is tool-CLI-specific: it commits to a stderr stream and a no-prefix format that is appropriate for one-shot CLI scripts, not for library code embedded inside a longer-running model. Keeping it under `neurox/tools/` makes the scope visible.

See also:

- `docs/dev/modules/tools/README.md`
- `docs/dev/modules/tools/calculate_1t1r_states.md`
