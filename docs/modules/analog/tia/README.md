# TIA Family

The TIA family provides BL clamp drivers for array solvers.

Rules:

- `TIAConfig` is the family base config
- each concrete implementation has its own config subclass
- `TIA.from_config(...)` dispatches by concrete config type
- concrete TIA configs may carry owned device configs and device design parameters

The current concrete implementation is `OpAmpTIA`.
