"""ADC submissions retain detached snapshots of each input field."""

from __future__ import annotations

import torch

from neurox.primitive.analog import AdcProber


def test_compiled_submissions_preserve_inputs_after_callers_modify_them(device) -> None:
    prober = AdcProber(sync_device=device)

    @torch.compile(dynamic=False, fullgraph=True)
    def submit(current, positive, negative):
        prober.submit_current(current)
        prober.submit_diff_voltage(v_pos__V=positive, v_neg__V=negative)

    expected = torch.arange(6.0, device=device).reshape(3, 2)
    inputs = expected.clone().requires_grad_()
    with prober:
        submit(*inputs.unbind())
        with torch.no_grad():
            inputs.fill_(-1)

    current_record, voltage_record = prober.result
    retained = (current_record.i_in__uA, voltage_record.v_pos__V, voltage_record.v_neg__V)
    for actual, original in zip(retained, expected.unbind(), strict=True):
        assert not actual.requires_grad
        torch.testing.assert_close(actual, original)
