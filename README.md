# CLIPTER MODEL

Lightweight CLIP + DETR training/inference project.

## 1) Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install torch torchvision tqdm pyyaml pillow
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

## 8) Git workflow

The repo tracks code/config only. Dataset files in `data/` remain local.

```powershell
git add .
git commit -m "Describe your change"
git push
```
