# CLIPTER MODEL

Lightweight CLIP + DETR training/inference project.

## 1) Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

For Colab, use the dedicated bootstrap instead of ad-hoc installs:

```bash
bash scripts/colab_bootstrap.sh
```

CLIPTER detector training does not require the OpenAI `clip` package. That AeroMixer failure path is not part of this repo's detector stack.

## 2) Dataset layout (local only)

`data/` is intentionally ignored by Git.

Expected root from `clipdetr/config.py`:

```text
data/weak-2/final_dataset/
  data.yaml
  stratified_train/
    images/
    labels/
  stratified_val/
    images/
    labels/
  stratified_val_10pct/
    images/
    labels/
```

## 3) Quick sanity check

```powershell
python dataset_smoke.py
```

## 4) Train detector (fast debug run)

```powershell
python clipdetr/train_detect.py --fast --subset 512
```

Colab instructions are in [docs/COLAB.md](docs/COLAB.md).

One-command end-to-end pipeline (recommended):

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
  --tune-thresholds `
  --benchmark `
  --yolo-metrics experiments/eval_yolo.json `
  --detr-metrics experiments/eval_detr.json
```

This runs train -> eval -> optional threshold tuning -> optional baseline benchmark.
It writes a reproducibility manifest to `<output-dir>/pipeline_manifest.json`.

For tiny-object datasets, enable tiled train/eval with stitched full-image metrics:

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

The pipeline builds a tiled dataset under `<output-dir>/prepared/`, trains on tiles, and stitches tiled validation predictions back into original-image coordinates for final metrics.

YOLO-style single entrypoint (from `data.yaml`):

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

This wrapper auto-resolves dataset root + train/val split names from `data.yaml` and launches `clipdetr/train_detect.py`.

Outputs:
- checkpoints in `checkpoints/`
- final model `light_detr_fast.pth`
- best-by-loss checkpoint `checkpoints/light_detr_best_loss_fast.pth`
- best-by-mAP checkpoint `checkpoints/light_detr_best_map_fast.pth`
- run logs in `experiments/` (config, per-epoch CSV, summary JSON)

## 5) Run detector inference

```powershell
python clipdetr/predict_detect.py `
  --checkpoint light_detr_fast.pth `
  --input data/weak-2/final_dataset/stratified_val/images `
  --output-dir predictions_test `
  --max-images 50 `
  --conf-thres 0.25 `
  --nms-iou 0.5
```

## 6) Full detector training

```powershell
python clipdetr/train_detect.py
```

Final output model:
- `light_detr.pth`

Current training defaults:
- image backbone starts from pretrained torchvision weights (`image_pretrained=True`)
- strict train/val class consistency is enabled (`strict_class_check=True`)
- CLIP-init is auto-discovered if one of these exists: `clip_backbone*.pth` (repo root or `checkpoints/`)
- train augmentation now includes box-aware geometric transforms (random crop + horizontal flip)

Useful runtime overrides:
- `--device auto|cpu|cuda`
- `--image-backbone mobilenet_v3_small|convnext_tiny`
- `--image-size <int>`
- `--embed-dim <int>`
- `--strict-class-check` (fail fast if val has classes missing in train)
- `--warmup-epochs <int>`
- `--grad-clip-norm <float>`
- `--use-ema --ema-decay <float>`

Example stronger training recipe:

```powershell
python clipdetr/train_detect.py `
  --data-root "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
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

Experimental multi-scale memory run (next architecture step):

```powershell
python scripts/pipeline.py `
  --mode run `
  --data "C:\path\to\final_dataset\data.yaml" `
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

This fuses multiple backbone stages into the detector memory so CLIPTER can be compared directly against the previous single-scale baseline.

## 7) Optional CLIP pretrain -> detector init

```powershell
python clipdetr/train_clip.py --fast
python clipdetr/train_detect.py --clip-init clip_backbone_fast.pth
```

If you omit `--clip-init`, training will still auto-pick a checkpoint when available unless `--no-auto-clip-init` is set.

## 8) Evaluate mAP from a checkpoint

```powershell
python clipdetr/utils/eval_detect.py `
  --checkpoint checkpoints/light_detr_best_map_fast.pth `
  --batch-size 16 `
  --output-json experiments/eval_best_map_fast.json
```

Tune postprocess thresholds on validation split:

```powershell
python clipdetr/utils/tune_detect_thresholds.py `
  --checkpoint checkpoints/light_detr_best_map_fast.pth `
  --batch-size 16 `
  --conf-grid 0.001,0.01,0.05,0.1,0.2,0.3 `
  --nms-grid 0.0,0.3,0.5 `
  --topk-grid 50,100 `
  --optimize map `
  --tile-stitch-eval `
  --output-json experiments/threshold_sweeps/best_thresholds.json
```

## 9) Hyperparameter sweep (lr, batch size, queries, decoder depth)

```powershell
python clipdetr/utils/sweep_detect.py `
  --fast `
  --subset 512 `
  --lrs 1e-4,5e-5 `
  --batch-sizes 8,16 `
  --num-queries 50,100 `
  --decoder-layers 2,3
```

Sweep reports are saved under `experiments/sweeps/...`.

## 10) Formal scratch vs CLIP-init comparison

```powershell
python clipdetr/utils/compare_clip_init.py `
  --clip-init clip_backbone_fast.pth `
  --fast `
  --subset 512
```

Comparison report is saved under `experiments/comparisons/...`.

## 11) Git workflow / repo hygiene

The repo tracks code/config only. Local artifacts are ignored:
- `data/`
- `checkpoints/`
- `experiments/`
- `predictions_*`

```powershell
git add .
git commit -m "Describe your change"
git push
```

## 12) Research plan execution (GPU later)

Plan files are in `configs/research/`.

Run stage-1 sanity plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/stage1_sanity.json `
  -DataRoot "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
  -DataYaml data.yaml `
  -TrainSplit train `
  -ValSplit valid `
  -Device auto `
  -ClipInit clip_backbone_fast.pth
```

If `use_clip_init=true` in a plan and `-ClipInit` is missing or invalid, the runner falls back to `--auto-clip-init` automatically.

Run stage-2 full plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/stage2_full.json `
  -DataRoot "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
  -DataYaml data.yaml `
  -TrainSplit train `
  -ValSplit valid `
  -Device auto `
  -ClipInit clip_backbone_fast.pth
```

Run strong recipe preset:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/strong_recipe_v1.json `
  -DataRoot "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
  -DataYaml data.yaml `
  -TrainSplit train `
  -ValSplit valid `
  -Device auto
```

Aggregate experiment reports:

```powershell
python clipdetr/utils/aggregate_detect_results.py `
  --experiments-root experiments `
  --output-csv reports/detect_runs_flat.csv `
  --output-grouped-csv reports/detect_runs_grouped.csv
```

## 13) Dataset audit before long runs

Use this to verify class/split integrity before expensive training:

```powershell
python clipdetr/utils/audit_detect_dataset.py `
  --data-root "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
  --data-yaml data.yaml `
  --train-split train `
  --val-split valid `
  --output-json reports/dataset_audit.json
```

## 14) Custom dataset splitting (80/10/10, 70/20/10, etc.)

Use this to repartition a YOLO dataset into any train/val/test ratio you want.

Example: 80/10/10 from existing `train,valid,test` content:

```powershell
python -m clipdetr.utils.split_yolo_dataset `
  --root "C:\Users\tsake\OneDrive\Desktop\one drive things\OneDrive\final dastaset for the project\merged_dataset" `
  --source-splits train,valid,test `
  --split-names train,valid,test `
  --ratios 80,10,10 `
  --seed 42 `
  --overwrite
```

Example: 70/20/10:

```powershell
python -m clipdetr.utils.split_yolo_dataset `
  --root "C:\Users\tsake\OneDrive\Desktop\one drive things\OneDrive\final dastaset for the project\merged_dataset" `
  --source-splits train,valid,test `
  --split-names train,valid,test `
  --ratios 70,20,10 `
  --seed 42 `
  --overwrite
```

Notes:
- The splitter can auto-detect source layout (`<root>/images` or `<root>/<split>/images`).
- It stratifies by primary class by default.
- It writes/updates `data.yaml` in the output root unless `--no-write-yaml` is set.

## 15) Unified baseline comparison (CLIPTER vs YOLO vs DETR)

Use one canonical CSV/JSON report format across different model families:

```powershell
python scripts/run_baseline_benchmarks.py `
  --dataset aircraft_damage_v2_2 `
  --split valid `
  --clipter-metrics experiments/eval_clipter.json `
  --yolo-metrics experiments/eval_yolo.json `
  --detr-metrics experiments/eval_detr.json `
  --summary-csv reports/baseline_benchmarks.csv `
  --output-json reports/baseline_benchmarks.json
```

You can also pass `--clipter-cmd`, `--yolo-cmd`, `--detr-cmd` to run and parse in one step.
If you use `scripts/pipeline.py --benchmark`, CLIPTER metrics path is filled automatically from pipeline eval output.

## 16) Competitive gap plan

Gap analysis and end-to-end execution plan (where others are strong/weak and how CLIPTER should respond):

- `docs/COMPETITIVE_GAP_PLAN.md`

## 17) Archived utilities

Legacy helpers were moved to keep active paths clean:

- `clipdetr/utils/archive/create_val_split.py`
- `clipdetr/utils/archive/print_split_stats.py`
