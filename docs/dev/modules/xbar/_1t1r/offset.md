# `neurox/xbar/_1t1r/offset.py`

`Offset1T1RXbar` is the current offset-coded xbar implementation.

It owns:

- physical-shape regrouping and offset/reference-column semantics;
- one `CircuitCore1T1R`;
- one `ReadOut`;
- the xbar-level capability surface defined by the generic `Xbar` base.

Current construction rule:

- member configs live on the xbar config;
- the xbar constructs its owned `CircuitCore1T1R` and `ReadOut`;
- family dispatch happens inside the respective family bases.

The offset xbar is the place where logical layout becomes physical layout. The core below it remains encoding-agnostic.
