# Hardware-Aware Training (HAT)

## Why HAT

A float neural network and a quantised crossbar chip disagree on every forward pass.  Each crossbar layer loses precision three times — once in the activation ADC, once in the weight grid, once in the post-aggregation requantise.  Plain post-training quantisation pushes that accumulated error straight into the loss; networks deeper than a few layers collapse.

Hardware-aware training turns the hardware into a first-class citizen of the forward pass: the loss sees exactly the integer pipeline the deployed chip will run, while the optimiser sees a smooth float reference through an STE bridge.  The training recipe and the chip become inseparable — one cannot be changed without rerunning the other.

## Scope

NeuroX's HAT targets one chip per run, specified by a TOML file plus a choice of tile implementation (the physical 1T1R circuit solver, or its lossless ideal twin).  It replaces every supported layer in a float model with a hardware-backed counterpart, fine-tunes against the task loss, and emits a single checkpoint that can be evaluated back on any compatible chip backend without re-training.

It does **not** try to be a general-purpose quantisation framework.  The quantisation grid is dictated by the chip, not the algorithm — there is no knob to "pick 4-bit or 8-bit weights" at the operator level.

## Methodology

### Quantisation grid comes from the hardware

The chip TOML pins the activation and weight integer ranges: a binary word-line drives an unsigned 4-bit activation grid, a 4-state RRAM with signed-digit encoding gives a symmetric 7-level weight grid.  The training code reads these two ranges off the macro and uses them as-is.

The **float** meaning of each integer level — how wide an input range the grid covers, where the zero lies — is learned per-layer by running-min/max observers during a short calibration pass.  Asymmetric for activations (arbitrary mean, zero-point learned), symmetric for weights (zero-mean, per-output-channel scale).

### Calibrate once, then freeze

The observer lifecycle in NeuroX has two phases, not the three of textbook HAT:

1. **Calibrate.** Forward-only on real data, weights unchanged.  The observer EMAs settle against the float-initialised activation distribution.  A few dozen batches is enough on both example models.
2. **Freeze.** Quantisation parameters are locked.  Training begins, weights update via STE, the grid is static.

There is no "warmup with grad" phase because on the coarse output grids the hardware gives us (16 levels post-requantise), observer EMA updates during training collide with the STE-noisy outputs they are supposed to summarise — small fluctuations in `s_y` shift the requantise boundary, which shifts the next forward's output, which shifts the EMA again.  The loop is not self-damping; it blows up within a couple of epochs.  Freezing after a passive calibration breaks it cleanly.

There is no "BN align" phase because the target chip does not have batch-norm statistics to realign; the example models either lack BN entirely or use stateless LayerNorm.  For a future model that does have BN, folding BN into the preceding linear before HAT starts is the right call, and a post-training re-fold to account for learned weight drift is future work.

### Forward: hardware-value, float-gradient

Every HAT layer computes two things and returns one:

- A **float-reference forward** with fake-quantised input and weight.  This is the gradient path.
- A **hardware forward** through the integer MAC pipeline, using the same quantisation parameters and the same weight.  This is the value returned to the caller.

An STE combine (`y_float + (y_hw − y_float).detach()`) makes the model see real hardware effects (clamping, ADC codes, requantise rounding, circuit noise on the physical path) while the optimiser sees a smooth, well-behaved gradient.  No need for custom autograd rules.

### Knowledge distillation is expected, not optional

At the grids the hardware forces on us, vanilla cross-entropy alone plateaus below float accuracy by a wide margin — and for anything deeper than a few layers it does not converge at all.  A frozen float teacher with identical architecture is the supervising signal: hard-label CE for direct task gradient, temperature-softened KL over logits for distribution matching, optional intermediate-representation MSE for structural alignment (used on BERT's pooled `[CLS]`).  Weights on each term are exposed as knobs; the defaults in the Makefile are what produced the reference checkpoints.

### Checkpoint is chip-agnostic

The saved checkpoint stores the integer weights, folded bias, and requantise parameters in ideal-integer scale — no chip-specific rescale baked in.  At load time, the target chip's ADC rescale factor is folded into the buffers.  One training run, many possible deployment chips, as long as the integer grid and transcoder wiring match.

## What to Expect

### LeNet-like models (few layers, no normalisation between them)

HAT training on a small CNN under the bundled 1T1R recovers most but not all of float accuracy.  The per-layer quantisation error is compound but bounded; the optimiser finds weights that align to the coarse grid.  Expect a 5–6-point gap to float on MNIST and similarly small benchmarks.  Without distillation the gap widens by another 1–2 points; with it, the training is stable from epoch ~3 and plateaus by epoch ~20.

### Transformer-like models (deep, LayerNorm between every linear)

Here the first 2–3 epochs look like complete failure: validation accuracy hovers around chance, cross-entropy barely moves.  This is not divergence.  It is the network finding its way through a deeper quantisation bottleneck before the classifier's integer output range becomes discriminative.  Once it breaks through, accuracy climbs monotonically over the next 3–4 epochs and the best checkpoint lands in the later half of the schedule.

Killing training at epoch 2 because it "looks stuck" is a common mistake — budget at least 8–10 epochs under distillation.  If CE has not moved at all by epoch 4, the learning rate is too low for the depth; if the loss has spiked to thousands, it is too high.  The useful window is narrow and model-specific; the Makefile defaults are what survives for the example models.

Expect a 6–8-point gap to float on SST-2.  Without distillation, the student will not break through at all on the bundled 1T1R.

### Physical versus ideal evaluation

Evaluation on the physical 1T1R circuit solver should land within 1–2 points of the lossless ideal tile.  Larger gaps indicate either a noise sub-table in the chip TOML is producing more error than the HAT forward saw, or the checkpoint and chip TOML are mismatched.  HAT by default uses the ideal tile during training for speed; if deployment noise is severe enough to open a bigger gap, HAT on the physical tile (noise-aware training) is the remedy, at 3–5× the training wall-clock.

## Where HAT Cannot Help

At the bundled chip's 16-level ADC, per-sub-tile ADC quantisation is the dominant error source, amplified by the radix shift-add that reassembles multi-digit activations and by the accumulator that sums across tiles.  For large-fan-in layers (transformer hidden dimensions in the hundreds) this noise floor is what caps accuracy.  No amount of learning-rate or distillation tuning can lower that floor — it is set by the chip spec.

Improvements that require code changes in the simulator, not training hyperparameters:

- Smarter observer calibration (percentile-based, KL-minimising, or learnable scales co-trained with weights).  The current min/max EMA is outlier-biased and is the first place to tune.
- A wider post-requantise output grid decoupled from the activation grid.  Preserves one extra bit of inter-layer signal.

Improvements that require a new chip spec (a different TOML), not training changes:

- Finer ADC (more output codes per tile), which directly lowers the per-sub-tile noise floor.
- More activation or weight digits.

Both categories are explicitly out of scope for the HAT training code itself.  HAT's job is to squeeze the best accuracy out of whatever chip it was handed.
