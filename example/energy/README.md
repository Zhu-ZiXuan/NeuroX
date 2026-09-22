# Side-channel energy evaluation

Run from the repository root after installing the declared example dependencies with `uv sync --extra demo`. Checkpoints, datasets, and the selected project's code must be available locally.

```bash
uv run --no-sync python -m example.energy.evaluate --family example --model sorbet --preset xue2020jssc --device cuda:0 --output log/energy/sorbet_xue.json
uv run --no-sync python -m example.energy.evaluate --family hardware_comparable --model bert --preset ye2023jssc --device cuda:0 --output log/energy/bert_ye.json
uv run --no-sync python -m example.energy.evaluate --family lenet_sparse_adc --model lenet --dataset ucihar --preset xue2020jssc --device cuda:0 --output log/energy/sparse_lenet.json
uv run --no-sync python -m example.energy.evaluate --family soul_fullgrid_sparse_adc --model mlp --dataset ucihar --preset ye2023jssc --device cuda:0 --output log/energy/soul_mlp.json
```

The checkpoint model supplies every numerical output; NeuroX observes complete layer inputs and discards its own output. Each sample's logits must match inference without observation exactly. Attention matrix products, bias additions, and operations outside mapped linear/convolution operators are not billed. The active `example` CIM entry is `sorbet`. FP32 LeNet and BERT are disabled by their local `configs/model.toml` before checkpoint loading or hardware construction.

## Preparation and measurement

A discovery forward identifies logical operator positions, then restores the model state and RNG before hardware construction. Each position receives a distinct unit, even when positions share an original module or equal weights. Its complete weight matrix or kernel is programmed once before measurement.

SOUL operators are observed before reference tiling or im2col. NeuroX receives complete logical operators without an outer tiling or chunking loop. Conv1d adds a singleton spatial dimension; convolution time/batch axes flatten for the unit interface, with their original shapes retained for reporting.

The inference script collects static hardware data and opens one profiler context per batch. Units retain basic-operation energy and timing positions; application analysis groups circuits into units and reduces operation axes to dataset samples. `--num-samples` selects the evaluation size. Each selected sample forms one inference batch, including inputs without an explicit batch axis.

## Model assumptions

Each active evaluation directory owns its `configs/` files. `model.toml` declares logical precision profiles; `unit.toml` declares unit and digital circuit parameters; `macro.toml` overrides the bundled presets; `macro.policy.toml` declares nonidealities. The Xue/Ye files combine those local fragments into concrete linear/conv2d config and policy sections. The factory loads them with `from_file`; only source-layer convolution geometry and an explicit merge override are bound in code.

Binary weights use one weight digit. Xue keeps a 256x512 physical array with 4 lanes and 64 scans. Ye keeps a 64x128 array by increasing input capacity to 64, with one weight digit and one RSM column per input. Xue uses one native input bit for spikes and two for wider activations; Ye uses one. Unit files declare the remaining input slicing. Xue's overrides preserve the original LSB mirror/input gains and converter references. Fixed peripheral PPA values remain preset assumptions, not a new silicon characterization.

`--ideal-macro` selects ideal twins from these configurations: analog dynamic energy is excluded, and static PPA covers retained ideal/local hardware. Digital circuit costs and clock timing are explicit assumptions in local unit files. Validation and exploration retain their own original preset geometries.

Only already quantized integer levels are accepted; no implicit absmax quantization is applied. Sorbet/hardware_comparable declare W1A4. Sparse-ADC directories declare W1A1 for spikes and W1A8 for layers whose existing forward applies 8-bit activation quantization. Floating classifier weights and unquantized dense activations are reported as unsupported. Each Sorbet directory also declares whether its F.linear receives scaled or already-integer activation codes, preventing duplicate scale removal. Signed weights on the unsigned Ye macro use adjacent positive/negative outputs within each macro. Local differences precede phase and input-slice accumulation and complete in the same digital cycle. Ye reads the pair on one lane in consecutive scans; ideal mode retains this placement.

Signed inputs are unsupported. Preparation reports all observed unsupported layers in `<output>.unsupported.json` and exits unsuccessfully. A signed input first encountered in a later batch also raises. The evaluator preserves complete logical layers and their signs. Grouped and depth-wise convolutions use independent hardware per group. `--merge` / `--no-merge` override input-slot sharing within each group; omission retains the file value.

Area is counted once per allocated circuit. Static energy uses each unit's basic-operation working window, assuming sequential operations and power-off outside work. The evaluator infers no whole-model timing or cross-unit pipeline and applies no macro-count normalization.

## Results

JSON contains per-sample and per-operator energy, area, leakage, working and powered durations, and leading shapes. Powered duration defaults to working duration. `summarize_profile` accepts `powered_duration__ns={name: tensor}` in the collected layout and device, leaving raw working durations unchanged.

`<output>.profile.pt` stores raw `dict[str, ProfileItem]` data on CPU. Local static totals include physical replication and exclude child hardware. JSON identifies the model-local hardware configuration file. Load this locally generated file with `torch.load(..., weights_only=False)`.

## Campaign

Run all bundled model/dataset combinations with both presets:

```bash
uv run --no-sync python -m example.energy.campaign --output-dir log/energy/campaign
```

The campaign checks GPU utilization and free memory, uses at most one worker per GPU, saves cases separately, and retries CUDA out-of-memory failures after a cooldown. It reuses successful outputs when resuming; choose a new directory after code or configuration changes. Other failures are recorded in `failures.json` and produce a failing exit status.
