# Research Plans (CLIPTER)

This folder contains reproducible experiment plans for detector training.

## Files

- `stage1_sanity.json`: short sanity suite to validate settings quickly.
- `stage2_full.json`: full training suite for final model selection.

## Run A Plan

Use the shared PowerShell runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/stage1_sanity.json `
  -DataRoot "C:\Users\tsake\OneDrive\Desktop\full dataset\merged_dataset" `
  -DataYaml data.yaml `
  -TrainSplit train `
  -ValSplit valid `
  -Device cuda `
  -ClipInit clip_backbone_fast.pth
```

Dry run (print commands only):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_detect_plan.ps1 `
  -Plan configs/research/stage1_sanity.json `
  -DryRun
```

If a plan item has `"use_clip_init": true` and `-ClipInit` is missing or invalid, that item is skipped.

## Aggregate Results

```powershell
python clipdetr/utils/aggregate_detect_results.py `
  --experiments-root experiments `
  --output-csv reports/detect_runs_flat.csv `
  --output-grouped-csv reports/detect_runs_grouped.csv
```
