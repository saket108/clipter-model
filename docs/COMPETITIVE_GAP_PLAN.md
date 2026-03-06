# CLIPTER Competitive Gap Plan (End-to-End)

## 1) Current comparison status

What CLIPTER already compares today:
- `scratch` vs `clip-init` via `clipdetr/utils/compare_clip_init.py`
- hyperparameter sweeps via `clipdetr/utils/sweep_detect.py`
- threshold tuning via `clipdetr/utils/tune_detect_thresholds.py`
- run aggregation via `clipdetr/utils/aggregate_detect_results.py`

What was missing (now added):
- one unified baseline comparison report format across CLIPTER / YOLO / DETR:
  - `scripts/run_baseline_benchmarks.py`

## 2) Where others are strong vs weak

### YOLO family
Strengths:
- very strong defaults for detection and small-object recall
- high-speed inference and mature confidence/NMS behavior

Weaknesses CLIPTER can exploit:
- weaker explicit image-text alignment than CLIP-based initialization
- less flexible for set-based query diagnostics and ablations

### DETR family
Strengths:
- strong global set-prediction formulation
- clean assignment and end-to-end architecture

Weaknesses CLIPTER can exploit:
- can be slower to converge without strong priors
- usually needs careful tuning on smaller custom datasets

### AeroMixer (internal reference)
Strengths to borrow:
- pipeline discipline: manifest, benchmark append, guardrails
- stronger experiment reproducibility conventions

Weaknesses CLIPTER can avoid:
- framework complexity overhead when only object detection is needed

## 3) CLIPTER target positioning

Primary objective:
- maximize custom-dataset detection mAP while preserving practical run simplicity

Differentiators:
- CLIP-init support with lightweight DETR head
- strict class/split checks and dataset auditing
- threshold tuning and structured experiment summaries

## 4) End-to-end execution plan

### Phase A: data quality gate (mandatory)
1. Split dataset reproducibly (`split_yolo_dataset.py`).
2. Audit class/split integrity (`audit_detect_dataset.py --strict`).
3. Fail run on leakage/class mismatch.

### Phase B: model baselines
1. Train CLIPTER strong recipe (`convnext_tiny`, EMA, strict checks, augmentation).
2. Train YOLO baseline on same split.
3. Train DETR baseline on same split.
4. Evaluate all on same validation split.
5. Aggregate with `scripts/run_baseline_benchmarks.py`.

### Phase C: gap-driven improvements
If CLIPTER trails YOLO on small objects:
- increase image size / apply tiling strategy
- tune query count/decoder depth
- tune thresholds and top-k/NMS
- rebalance class distribution and clean tiny-label noise

If CLIPTER trails DETR on AP75:
- tighten localization recipe (longer schedule, stronger augmentation, clip grad)
- inspect bbox/GIoU trends in summaries

### Phase D: release-quality process
1. keep one canonical strong recipe JSON in `configs/research/`
2. keep one benchmark CSV for all models
3. lock final thresholds from tuning JSON
4. record best checkpoint + exact command + split seed

## 5) Acceptance criteria

- dataset audit passes with zero critical errors
- CLIPTER run reproducible from one command + one config
- baseline comparison CSV contains CLIPTER/YOLO/DETR rows for same split
- tuned-threshold CLIPTER improves over untuned CLIPTER on `map` and `map50`
- final report includes:
  - best checkpoint path
  - best thresholds
  - comparative table
