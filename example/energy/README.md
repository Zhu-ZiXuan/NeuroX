# Side-channel energy evaluation

Run from the NeuroX repository root with its uv environment and the `demo` and `benchmark` dependency extras:

```bash
uv run --no-sync python -m example.energy.evaluate --family example --model lenet --preset xue2020jssc --device cuda:0 --output log/energy/lenet_xue.json
uv run --no-sync python -m example.energy.evaluate --family hardware_comparable --model bert --preset ye2023jssc --device cuda:0 --output log/energy/bert_ye.json
uv run --no-sync python -m example.energy.evaluate --family lenet_sparse_adc --model lenet --dataset ucihar --preset xue2020jssc --device cuda:0 --output log/energy/sparse_lenet.json
uv run --no-sync python -m example.energy.evaluate --family soul_fullgrid_sparse_adc --model mlp --dataset ucihar --preset ye2023jssc --device cuda:0 --output log/energy/soul_mlp.json
```

The original checkpoint model supplies every numerical output. NeuroX runs independently on the observed operator operands. Every evaluated sample is checked against inference without observation; logits must match exactly. Grouped convolutions use one unit per group, and one-dimensional convolutions lower to height-one two-dimensional convolutions. Attention matrix products, bias additions, and other operations outside the mapped matrix products are not billed.

Circuit configurations come from the bundled Xue2020 and Ye2023 presets. Execution uses their ideal macro twins, so reported energy covers configured digital operations and excludes analog dynamic energy. Input-tile recovery is functional only and contributes no accumulator energy. Digital input-phase accumulator and shift-adder costs are evaluation assumptions of 50 and 80 fJ per operation, respectively. These results validate integration and are not physical macro energy estimates.

Bounded integer operands retain their codes. Generic floating operands use symmetric per-tensor quantization with eight magnitude bits. Quantized BERT linear operators retain their activation codes and binary weight signs; SOUL spike operators retain one-bit activations and binary weights. Signed inputs use separate positive and negative accesses; the unsigned Ye weight carrier also uses separate positive and negative weight planes. The original model is unaffected by these measurement representations.

`--merge` enables input-slot sharing; the default gives each logical output tile its own macro. `--num-samples` selects the evaluation size (one by default). `--batch-chunk` bounds simultaneous caller rows, including flattened time and batch axes (one by default). `--spatial-chunk` bounds convolution output positions per call while preserving kernel halos (32 by default). Each hardware call uses a fresh profiler context.

The example family accepts `lenet`, `bert`, and `sorbet`. LeNet and BERT load the original floating-point model state dictionaries. Checkpoints and datasets must already be present locally.

To check all bundled model/dataset combinations with both presets:

```bash
uv run --no-sync python -m example.energy.campaign --output-dir log/energy/campaign
```

The campaign checks GPU utilization and free memory before launch, uses at most one worker per GPU, saves each case separately, and retries CUDA out-of-memory failures after one hour. Successful output files are reused when resuming the same campaign directory; use a new directory after code or configuration changes. Other failures are recorded in `failures.json` and cause a failing exit status.
