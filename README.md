# CLIPTER MODEL

Lightweight CLIP + DETR training/inference project.

## 1) Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install torch torchvision tqdm pyyaml pillow scipy
```

If you use extra dependencies in your local code, install those as needed.

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

## 7) Optional CLIP pretrain -> detector init

```powershell
python clipdetr/train_clip.py --fast
python clipdetr/train_detect.py --clip-init clip_backbone_fast.pth
```

## 8) Evaluate mAP from a checkpoint

```powershell
python clipdetr/utils/eval_detect.py `
  --checkpoint checkpoints/light_detr_best_map_fast.pth `
  --batch-size 16 `
  --output-json experiments/eval_best_map_fast.json
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
