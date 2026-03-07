# CLIPTER Detector Redesign Plan

This document defines the next CLIPTER detector iteration after the confirmed v3 baseline on `aircraft-skin-defects-new-dataset.v3-best.yolov8`.

It is not a general roadmap. It is the concrete plan for the next model cycle.

## 1. Locked Baseline

Current official baseline:

- dataset: `aircraft-skin-defects-new-dataset.v3-best.yolov8`
- backbone: `convnext_tiny`
- image size: `320`
- detector: single-scale `LightDETR`
- best validation metrics:
  - `map`: `0.2754`
  - `map50`: `0.6066`
  - `map75`: `0.2285`

Observed AP50 by class:

- `paint-off`: `0.7449`
- `dent`: `0.6807`
- `crack`: `0.6252`
- `missing-head`: `0.5024`
- `scratch`: `0.4798`

Interpretation:

- detection is functional and credible
- weak point is localization, not coarse object discovery
- the `map50 -> map75` drop is too large
- threshold tuning is not the main bottleneck

## 2. What We Already Tested

### Baseline that worked

- `320`
- single-scale memory
- `convnext_tiny`
- 80 epochs

This is the current reference.

### Combined ablation that did not help

- `384`
- multi-scale memory enabled

Problem:

- changed resolution and architecture at the same time
- slower training
- early validation metrics were below the baseline

### Resolution-only ablation that also did not show an early win

- `384`
- single-scale memory

Problem:

- also below the `320` baseline at matched early checkpoints
- Colab GPU time ended before a full answer, but there was no early signal strong enough to justify continuing

Conclusion:

- do not spend more GPU on unconstrained ablations
- redesign the detector architecture first

## 3. Core Diagnosis

CLIPTER is currently limited by three things:

1. single-stream detector memory
2. weak multi-scale spatial handling compared with YOLO-style detectors
3. insufficient box refinement quality at stricter IoU thresholds

What this means in practice:

- increasing resolution alone is not enough
- concatenating backbone stages into one token memory is not enough
- the next gain must come from a more detection-native feature path

## 4. Design Direction

The next detector iteration should move toward:

- hierarchical backbone features
- explicit multi-scale fusion
- decoder input that preserves scale structure instead of flattening everything into one undifferentiated token bank

The target is not to turn CLIPTER into YOLO.
The target is to make CLIPTER detection-native enough that it stops losing obvious localization quality.

## 5. Next Architecture Work

### Phase 1: structured multi-scale memory

Replace the current simple concatenation strategy with a structured multi-scale path.

Requirements:

- expose multiple ConvNeXt stages explicitly
- project each stage to the detector hidden dimension
- keep per-level identity available to the decoder
- avoid mixing all scales too early

This can still feed the DETR-style decoder, but the decoder should receive scale-aware memory, not a flat bag of tokens.

### Phase 2: lightweight feature fusion neck

Add a small detection neck before decoder consumption.

Target behavior:

- fuse low/mid/high backbone features
- strengthen small and medium object representation
- preserve higher-resolution structure better than the current final-stage-only path

Acceptable options:

- minimal FPN-style top-down fusion
- lightweight PAN-style variant

Non-goal:

- do not build a huge detection framework rewrite
- keep the neck narrow and bounded so CLIPTER stays manageable

### Phase 3: box refinement bias

Improve localization quality directly.

Candidate directions:

- iterative box refinement across decoder layers
- stronger bbox/GIoU weighting schedule
- query initialization with stronger spatial priors

The next detector revision must target `map75`, not only `map50`.

## 6. Experiment Discipline

From this point forward:

- change one major variable at a time
- compare at matched checkpoints, not only final runs
- stop weak runs early when they are materially behind baseline

Required experiment order:

1. new structured multi-scale memory at `320`
2. new neck + structured memory at `320`
3. only then test `384` if the `320` redesign is already better

Do not repeat:

- `384` plus new architecture in one first-pass run

## 7. Acceptance Gates

An architecture change is worth keeping only if it clears both gates below.

### Gate A: early checkpoint gate

At matched early epochs, it must not materially trail the baseline on:

- validation loss
- `map`
- `map50`

### Gate B: final quality gate

It must beat the locked baseline on at least one of these in a meaningful way:

- `map`
- `map75`

Preferred outcome:

- `map` improvement with stable or improved `map50`
- smaller gap between `map50` and `map75`

## 8. Immediate Next Tasks

Do these before the next GPU run:

1. refactor `ImageEncoder` to return explicit named feature stages cleanly
2. add a small multi-scale fusion neck module
3. update `LightDETR` to consume structured scale-aware memory
4. add unit tests for:
   - stage extraction shapes
   - neck output shapes
   - detector forward pass with structured multiscale memory
5. add a single clean training flag for the redesign path

## 9. What Not To Do

Do not spend time on these first:

- more threshold sweeps
- more random image-size sweeps
- larger decoder depth without feature-path redesign
- more combined ablations that change several dimensions at once

Those are second-order changes. The main gap is architectural.

## 10. Success Condition

The redesign is successful if CLIPTER becomes better at strict localization while staying simple enough to train from one command on a YOLO dataset.

That means:

- baseline remains reproducible
- redesigned model stays operational in Colab
- the next best checkpoint beats the current `map=0.2754` baseline for the right reason, not from metric noise
