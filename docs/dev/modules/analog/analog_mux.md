# `neurox/analog/analog_mux.py`

`AnalogMux` is a leaf analog transport block.

It owns:

- its own design/spec config
- fabricate-time static state, if any
- per-call differential transport behaviour and energy

It is not polymorphic today, so parent circuits construct it directly from its config rather than dispatching through a family base.
