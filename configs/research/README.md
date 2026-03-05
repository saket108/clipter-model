# Research Plans (CLIPTER)

This folder contains reproducible experiment plans for detector training.

## Files

- `stage1_sanity.json`: short sanity suite to validate settings quickly.
- `stage2_full.json`: full training suite for final model selection.
- `strong_recipe_v1.json`: single strong training recipe for a high-quality baseline run.

## Run A Plan

Use the shared PowerShell runner:

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

Dry run (print commands only):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/stage1_sanity.json `
  -DryRun
```

If a plan item has `"use_clip_init": true` and `-ClipInit` is missing/invalid, the runner now falls back to `--auto-clip-init` instead of skipping.

## Run Strong Recipe

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/strong_recipe_v1.json `
  -DataRoot "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
  -DataYaml data.yaml `
  -TrainSplit train `
  -ValSplit valid `
  -Device auto
```

`strong_recipe_v1` enables strict class checks, geometric+photometric train augmentation, EMA, warmup, clipping, larger model capacity, and auto CLIP-init discovery.

## Aggregate Results

```powershell
python clipdetr/utils/aggregate_detect_results.py `
  --experiments-root experiments `
  --output-csv reports/detect_runs_flat.csv `
  --output-grouped-csv reports/detect_runs_grouped.csv
```
