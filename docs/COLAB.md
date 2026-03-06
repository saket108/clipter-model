# Colab Setup

This repo does not require the OpenAI `clip` package for detector training. The Colab path is simpler than AeroMixer:

1. Clone the repo.
2. Run the bootstrap script.
3. Extract your YOLO dataset zip.
4. Launch `scripts/pipeline.py`.

## Minimal detector-only setup

```bash
%cd /content
!git clone https://github.com/saket108/clipter-model.git clipter
%cd /content/clipter
!bash scripts/colab_bootstrap.sh
```

## Optional text / CLIP-pretraining setup

Use this only if you also want `train_clip.py` or tokenizer-backed caption paths:

```bash
!bash scripts/colab_bootstrap.sh --with-text
```

## Extract dataset

```bash
!mkdir -p /content/data/aircraft_skin
!unzip -q "/content/aircraft-skin-defects-new-dataset.v1-best.yolov11.zip" -d /content/data/aircraft_skin
!find /content/data/aircraft_skin -name data.yaml
```

## Run CLIPTER baseline

```bash
!python scripts/pipeline.py \
  --mode run \
  --data /content/data/aircraft_skin/data.yaml \
  --output-dir /content/output/clipter_aircraft_skin_v1 \
  --device auto \
  --epochs 80 \
  --batch-size 8 \
  --image-backbone convnext_tiny \
  --image-size 320 \
  --embed-dim 384 \
  --num-workers 2 \
  --tune-thresholds
```

## Run tiled version

```bash
!python scripts/pipeline.py \
  --mode run \
  --data /content/data/aircraft_skin/data.yaml \
  --output-dir /content/output/clipter_aircraft_skin_tiles_v1 \
  --device auto \
  --epochs 80 \
  --batch-size 8 \
  --image-backbone convnext_tiny \
  --image-size 320 \
  --embed-dim 384 \
  --num-workers 2 \
  --tile-size 640 \
  --tile-overlap 0.2 \
  --tile-min-cover 0.35 \
  --tile-stitch-eval \
  --tune-thresholds
```

## Save outputs to Drive

```bash
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/output /content/drive/MyDrive/clipter_outputs
```
