# Operator workflow

A library workflow selects a model, establishes its state, executes integer operands, and interprets its observations. The [quickstart](../../get_started/README.md) provides a complete ideal linear example.

## Select a model and value domain

The [unit API](../../api/units.md) offers ideal and CIM linear and convolution implementations. Ideal units provide integer arithmetic without analog execution. CIM units map operands onto a configured macro and recovery circuits. Read the [unit model](../../reference/architecture/unit/family.md) and the chosen components' assumptions before interpreting differences between them.

Provide operands in the unit's declared integer value domain. Preparing quantized inputs and weights from a neural network belongs to the calling application. Operator results and hardware statistics can be examined independently of a training pipeline.

## Construct the hardware

Construct a concrete unit in Python, or use the file factories in the [application API](../../api/python.md). [Configuration and policy](../../api/configuration.md) describe the hardware and the selected run behavior. Bundled presets supply reusable circuit designs; a unit configuration also supplies its mapping and digital parameters.

Set logical weight shape and any convolution geometry explicitly. Configuration fields and factory signatures are documented from source; geometry and value-domain checks belong to those interfaces.

## Establish state

Move the assembled unit to the selected device, establish temperature, and fabricate it when the model requires a fabricated realization. Program weights and optional bias before execution. Place operand tensors on the device expected by the unit. Follow the owning interface's dtype and shape requirements.

[Physical state](../../system_design/physical_state.md) explains which changes require a new fabrication or programming event. Repeated execution can retain the same hardware realization while sampling fresh access variation.

## Collect observations

Set the profile leading rank on the assembled unit to match the independent operation axes in its input. Collect static data after physical-state setup, then open a profiler context around the operation. Keep the assembled hardware and names fixed during measurement.

Construct a reporter after the context exits. The [profiler and reporter API](../../api/python.md) defines the retained fields, concatenation, and reporting options. [PPA accounting](../../system_design/ppa_accounting.md) explains local hardware ownership, working and powered durations, and which aggregation belongs to the application.

## Check the result

Compare numerical outputs against a suitable ideal calculation under the same value-domain assumptions. For physical PPA claims, consult [validation campaigns](../../validation/campaigns.md) and preserve the chosen configurations, dependency versions, device, and measurement basis. A matching arithmetic result alone does not validate a physical energy estimate.
