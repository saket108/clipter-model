# CLIPTER

CLIPTER is a lightweight object detection research repo built around YOLO-format datasets, modern image backbones, and a DETR-style detection head.

The repo is optimized for one practical workflow:
- train on a clean YOLO dataset
- evaluate with reproducible outputs
- tune thresholds
- compare against YOLO and DETR baselines
- iterate with tiled or multi-scale variants when the baseline shows a clear gap

## What CLIPTER Includes

- ConvNeXt-Tiny and MobileNetV3-Small image backbones
- Lightweight DETR-style decoder for box and class prediction
- One-command train/eval/tuning pipeline
- Tiled train/eval with stitched full-image metrics for small-object datasets
- Optional multi-scale decoder memory for stronger localization experiments
- Colab bootstrap with detector-only dependencies
- Benchmark aggregation and experiment manifests for reproducibility

## Recommended Entry Points

Use these scripts first:

- `scripts/pipeline.py`: end-to-end train, eval, threshold tuning, and optional benchmark logging
- `scripts/train_from_data_yaml.py`: simpler training wrapper from a YOLO `data.yaml`
- `clipdetr/train_detect.py`: low-level detector training entrypoint
- `scripts/run_baseline_benchmarks.py`: unified CLIPTER vs YOLO vs DETR reporting

## Repository Layout

```text
clipdetr/
  models/              core detector and backbone code
  datasets/            YOLO dataset loading
  utils/               evaluation, threshold tuning, audits, splitting
scripts/
  pipeline.py          recommended end-to-end runner
  train_from_data_yaml.py
  run_baseline_benchmarks.py
  colab_bootstrap.sh
configs/
  research/            staged experiment plans
  presets/             higher-level run presets
```

## Installation

### Local

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
python scripts/verify_env.py
```

### Colab

```bash
bash scripts/colab_bootstrap.sh
python scripts/verify_env.py
```

Detailed Colab notes are in [docs/COLAB.md](docs/COLAB.md).

## Dataset Contract

CLIPTER expects a standard YOLO dataset with a `data.yaml` and split folders.

```text
dataset_root/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

The split names can differ if `data.yaml` points to them correctly.

Before long training runs, audit the dataset:

```powershell
python clipdetr/utils/audit_detect_dataset.py `
  --data-root "C:\path\to\dataset_root" `
  --data-yaml data.yaml `
  --train-split train `
  --val-split valid `
  --output-json reports/dataset_audit.json
```

If you need a custom split such as `80/10/10` or `70/20/10`, use:

```powershell
python -m clipdetr.utils.split_yolo_dataset `
  --root "C:\path\to\dataset_root" `
  --source-splits train,valid,test `
  --split-names train,valid,test `
  --ratios 80,10,10 `
  --seed 42 `
  --overwrite
```

## Quick Start

### 1. Run the baseline pipeline

This is the default workflow for a serious run.

```powershell
python scripts/pipeline.py `
  --mode run `
  --data "C:\path\to\dataset\data.yaml" `
  --output-dir experiments/pipeline_runs/run1 `
  --device auto `
  --epochs 80 `
  --batch-size 8 `
  --image-backbone convnext_tiny `
  --image-size 320 `
  --embed-dim 384 `
  --tune-thresholds
```

Pipeline outputs are written under the selected `--output-dir`:

- `train_summary.json`
- `eval_clipter.json`
- `best_thresholds.json`
- `pipeline_manifest.json`

### 2. Use the direct training wrapper

Use this when you want a simpler YOLO-style training entrypoint without the full pipeline wrapper.

```powershell
python scripts/train_from_data_yaml.py `
  --data "C:\path\to\dataset\data.yaml" `
  --device auto `
  --epochs 80 `
  --batch-size 8 `
  --image-backbone convnext_tiny `
  --image-size 320 `
  --embed-dim 384 `
  --tag strong_recipe_v1
```

### 3. Use the low-level trainer

Use this only when you want full control over detector arguments.

```powershell
python clipdetr/train_detect.py `
  --data-root "C:\path\to\dataset_root" `
  --data-yaml data.yaml `
  --train-split train `
  --val-split valid `
  --device auto `
  --image-backbone convnext_tiny `
  --image-size 320 `
  --embed-dim 384 `
  --batch-size 8 `
  --epochs 80 `
  --lr 8e-5 `
  --weight-decay 1e-4 `
  --warmup-epochs 5 `
  --grad-clip-norm 1.0 `
  --use-ema `
  --ema-decay 0.999 `
  --strict-class-check `
  --tag strong_recipe_v1
```

## Evaluation and Threshold Tuning

Evaluate a checkpoint directly:

```powershell
python clipdetr/utils/eval_detect.py `
  --checkpoint checkpoints/light_detr_best_map_fast.pth `
  --batch-size 16 `
  --output-json experiments/eval_clipter.json
```

Tune validation thresholds:

```powershell
python clipdetr/utils/tune_detect_thresholds.py `
  --checkpoint checkpoints/light_detr_best_map_fast.pth `
  --batch-size 16 `
  --conf-grid 0.001,0.01,0.05,0.1,0.2,0.3 `
  --nms-grid 0.0,0.3,0.5 `
  --topk-grid 50,100 `
  --optimize map `
  --output-json experiments/threshold_sweeps/best_thresholds.json
```

## Small-Object Workflow

For small or sparse defects, use tiled training and stitched full-image evaluation.

```powershell
python scripts/pipeline.py `
  --mode run `
  --data "C:\path\to\dataset\data.yaml" `
  --output-dir experiments/pipeline_runs/run_tiles `
  --device auto `
  --epochs 80 `
  --batch-size 8 `
  --image-backbone convnext_tiny `
  --image-size 320 `
  --embed-dim 384 `
  --tile-size 640 `
  --tile-overlap 0.2 `
  --tile-min-cover 0.35 `
  --tile-stitch-eval `
  --tune-thresholds
```

This builds a tiled dataset under the pipeline output directory, trains on tiles, and stitches validation predictions back into original-image coordinates for final metrics.

## Multi-Scale Architecture Experiment

CLIPTER now supports an optional multi-scale decoder memory path. Use this when the baseline detects objects but localization is still weak.

```powershell
python scripts/pipeline.py `
  --mode run `
  --data "C:\path\to\dataset\data.yaml" `
  --output-dir experiments/pipeline_runs/multiscale_v1 `
  --device auto `
  --epochs 80 `
  --batch-size 8 `
  --image-backbone convnext_tiny `
  --image-size 384 `
  --embed-dim 384 `
  --use-multiscale-memory `
  --multiscale-levels 3 `
  --tune-thresholds
```

## Baseline Comparison

Use one report format across model families:

```powershell
python scripts/run_baseline_benchmarks.py `
  --dataset aircraft_skin_defects `
  --split valid `
  --clipter-metrics experiments/eval_clipter.json `
  --yolo-metrics experiments/eval_yolo.json `
  --detr-metrics experiments/eval_detr.json `
  --summary-csv reports/baseline_benchmarks.csv `
  --output-json reports/baseline_benchmarks.json
```

If you run `scripts/pipeline.py --benchmark`, the CLIPTER metrics path is filled automatically.

## Advanced and Research Workflows

Formal staged experiments are kept under `configs/research/` and are launched with `scripts/run_detect_plan.ps1`.

Example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/strong_recipe_v1.json `
  -DataRoot "C:\path\to\dataset_root" `
  -DataYaml data.yaml `
  -TrainSplit train `
  -ValSplit valid `
  -Device auto
```

Aggregate experiment reports with:

```powershell
python clipdetr/utils/aggregate_detect_results.py `
  --experiments-root experiments `
  --output-csv reports/detect_runs_flat.csv `
  --output-grouped-csv reports/detect_runs_grouped.csv
```

## Additional Documentation

- [docs/COLAB.md](docs/COLAB.md): Colab setup and runtime guidance
- [docs/COMPETITIVE_GAP_PLAN.md](docs/COMPETITIVE_GAP_PLAN.md): model comparison and improvement roadmap

## Repository Hygiene

Tracked content is code and configuration only. Local artifacts should remain untracked:

- `data/`
- `checkpoints/`
- `experiments/`
- `predictions_*`

Legacy helpers that are no longer active entrypoints were moved into `clipdetr/utils/archive/`.
