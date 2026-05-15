# Example Pipelines

The examples demonstrate the full NeuroX flow — float training, hardware-aware training under a real chip spec, and evaluation against both an ideal tile and the physical 1T1R circuit solver — on four workloads spanning very different depths and structures:

- A small CNN (LeNet-5) on MNIST.
- A transformer encoder (BERT-small) fine-tuned on SST-2.
- A spiking CNN (SpikingVGG-5) on CIFAR-10, built on the external [Soul](https://github.com/) SNN library.
- A spiking transformer (Spikformer-256) on CIFAR-10, also built on Soul.

They are not about the specific networks.  They are about showing how much accuracy you can keep when a model is pushed through a real crossbar chip, and what hyperparameter regime gets you there.

The two SNN examples (SpikingVGG, Spikformer) embed Soul's spiking neurons (LIFNode) and forward-time loop unchanged — the example code only adds a thin direct-coding wrapper that turns the standard `(B, C, H, W)` dataloader output into the `(T, B, C, H, W)` Soul vision models expect.  All replaceable layers (`nn.Conv2d`, `nn.Linear`, with their sibling `BatchNorm` folded in) are swapped for crossbar operators by the standard `replace_for_hat` / `build_evaluator` pipeline; LIF neurons, MaxPool, LayerNorm, and the attention softmax stay in float.

## What the Reference Runs Show

Under the bundled 1T1R configuration (4-state RRAM, binary word-line, 16-level ADC):

| Model                   | Replaceable layers   | Float   | HAT on ideal tile | HAT on physical tile |
| ----------------------- | -------------------- | ------- | ----------------- | -------------------- |
| LeNet-5 / MNIST         | 2 Conv + 3 Linear    | ~99 %   | ~93 %             | ~92 %                |
| BERT-small / SST-2      | 26 Linear            | ~83 %   | ~76 %             | ~76 %                |
| SpikingVGG-5 / CIFAR-10 | 4 Conv + 1 Linear    | 86.1 %  | 55.0 %            | 53.7 %               |
| Spikformer-256 / CIFAR-10 | 5 Conv + 13 Linear | 87.3 %  | 20.0 %            | 21.1 %               |

Three lessons from these numbers:

- The gap to float is **moderate on shallow networks** (a few points), **much larger on deep networks where the input distribution is wide-range**.  LeNet's input is binary-ish digits and BERT's wide-range float embeddings still happen to compress acceptably onto the per-tile 16-level ADC, but SpikingVGG-5 and Spikformer-256 — whose pre-LIF activations span a much wider range that the LIF threshold fires sparsely on — lose substantial accuracy.  Spikformer-256's 18-macro depth sits squarely in the regime where 16-level ADC error compounds across layers and HAT cannot recover it (see `MEMORY.md` and the corresponding ADC-resolution discussion); a higher-resolution chip TOML is needed to train these SNNs to anywhere close to float.
- The gap between the ideal tile and the physical tile is **small** (≲ 2 points) on every example, including the SNNs.  Most of the accuracy loss comes from the chip's digital grid (ADC resolution, weight states, activation bit-width), not from analog non-idealities.  Noise-aware training on the physical tile closes the remaining gap but is 3–5× slower.
- Physical evaluation of the SNNs is sample-rate-bound: ~1.8 s/sample for SpikingVGG-5 (5 macros × T=4) and ~10 s/sample for Spikformer-256 (18 macros × T=4) on a single H100-class GPU.  The reference runs are limited to 2,000 (VGG) and 256 (Spikformer) test samples; the gap to a full-test-set number is dominated by sampling noise, not chip noise.

## Expected Training Dynamics

### Shallow CNN

Training is well-behaved from the start.  Validation accuracy climbs smoothly and plateaus well before the end of a long schedule.  The saved checkpoint is the peak of that curve, not the final epoch — late epochs often lose a point to overfitting against the fake-quant gradient.

A high learning rate spikes the loss within a few epochs; a low one just converges slowly to the same plateau.  The useful range is narrow and is where the defaults sit.

### Deep Transformer

Expect 2–3 epochs of apparent stall at chance accuracy.  Cross-entropy barely moves, validation accuracy oscillates around random guessing, and the model looks completely broken.  This is normal.  The cosine learning-rate schedule needs that many epochs to warm the 26-layer integer pipeline enough for the classifier's output range to become discriminative.  Once it breaks through — usually at epoch 4 — accuracy climbs fast, peaks in the second half of the schedule, and the best checkpoint is whatever the run tracked as best-val-acc.

If CE is still equal to `ln(num_classes)` by epoch 5, the learning rate is too low for the depth.  If the loss has spiked to thousands at any point, it is too high.  Somewhere in between there is a working setting; the Makefile defaults pin it for the reference model.

### Knowledge distillation

A frozen float teacher supervises the quantised student on both examples.  This is not cosmetic — on a 16-level output grid, pure cross-entropy stalls below 91 % on MNIST and below chance on SST-2.  The teacher's soft logits plus (for BERT) an intermediate-representation MSE provide gradient direction that the hardware pipeline can follow.  The Makefile exposes the KD weights; tuning them sensibly is worth a point or two, tuning them wildly costs many.

## Checkpoint Handling

One checkpoint per model is canonical.  It is overwritten each time training succeeds, with the peak validation accuracy during that run, not the last epoch.  The checkpoint is written in a chip-agnostic form — evaluation against any compatible chip TOML reproduces the intended hardware behaviour without retraining.  Swapping between the ideal tile and the physical tile is a single knob at evaluation time; accuracy, timing, and energy metrics all come out of the same run.

## When to Override Defaults

The Makefile targets are tuned to reproduce the reference checkpoints with no arguments.  Legitimate reasons to override:

- **Different GPU.** Pass the device explicitly.  Running two targets on different devices in parallel is the fastest way to iterate on both models at once.
- **Shorter schedule for a smoke test.** Cut epochs to confirm the pipeline wires together; the early epochs will look worse than the reference on both models, especially the transformer.
- **Sweeping one knob at a time.** Write the result to a scratch checkpoint path, compare evaluations, swap into the canonical path only if it is better.  Changing several knobs at once makes the outcome impossible to attribute.
- **New chip TOML.** A different ADC resolution or weight state count invalidates the tuned hyperparameters.  Start from the reference settings, re-calibrate expectation from first principles (how many bits per layer? how deep? does distillation apply?), and iterate from there.

Extending the search past what the Makefile exposes — a different optimiser, a different LR schedule, mixed precision across layers — is a code change, not a hyperparameter sweep.  See the HAT design document for what is fixed by the training recipe and what is free.

## Troubleshooting the Gap to Float

A few percentage points below float is expected and documented above.  A much larger gap points to one of a small number of causes:

- **Validation on a chip configuration that does not match training.**  The integer grid (activation bit-width, weight levels) must be consistent between the chip TOML used for HAT and the one used for evaluation.  Different configs cannot read the same checkpoint.
- **Observer calibration not settled.**  Too few calibration batches leaves the quantisation parameters biased by the first batch's outliers.  Symptom: early epochs of HAT have a very large initial loss that comes down faster than usual.
- **LR too high or too low for the depth.**  See the training-dynamics notes above.

A gap that is fundamental to the chip — not a training problem — manifests as a training curve that cleanly plateaus and stays flat across different hyperparameter settings.  At that point the chip spec is the remaining lever.

## Related Documents

- Design of the training recipe and why it differs from textbook HAT: `doc/design/hat_training.md`.
- How the code is organised and which directory owns which concern: `doc/design/architecture.md`.
- Physical 1T1R circuit solver: `doc/design/solver_1t1r.md`.
