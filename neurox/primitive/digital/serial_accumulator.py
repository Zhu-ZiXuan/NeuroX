"""Digital modular-arithmetic accumulator for a time-serial operand stream.

See also:
    docs/reference/primitive/digital/serial_accumulator.md
"""

from .accumulator import Accumulator


class SerialAccumulator(Accumulator):
    """Modular accumulator whose reduced axis is one register's arrival stream.

    The fold, the modular wrap and the per-operand billing are the
    accumulator's; only the realization of the reduced axis differs —
    successive arrivals on a single register per instance, rather than legs of
    an adder tree.
    """
