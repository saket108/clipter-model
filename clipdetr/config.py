"""Project configuration for clipdetr (minimal, editable)
"""
from dataclasses import dataclass

@dataclass
class Config:
    # model / data
    image_size: int = 224
    max_text_len: int = 32

    # lightweight defaults (close to YOLO-n scale parameter budget)
    image_backbone: str = "mobilenet_v3_small"
    image_pretrained: bool = False
    embed_dim: int = 288     # shared encoder width
    proj_dim: int = 256      # final projection dim (CLIP space)
    vocab_size: int = 2048   # capped text vocab for a compact embedding table
    text_num_heads: int = 8
    text_num_layers: int = 2
    text_ff_dim: int = 576
    text_dropout: float = 0.1

    # optimization
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 5

    # contrastive / scaling
    temperature: float = 0.07

    # misc
    seed: int = 42
    num_workers: int = 4

    # dataset (set these before training)
    data_root: str = "data/weak-2/final_dataset"        # dataset root containing split folders
    data_yaml: str = "data.yaml"                        # YOLO data.yaml relative to data_root or absolute path
    annotation_format: str = "auto"                     # one of ["auto", "yolo", "coco_json"]
    train_annotations: str | None = None                # optional JSON annotations path for train split
    val_annotations: str | None = None                  # optional JSON annotations path for val split
    test_annotations: str | None = None                 # optional JSON annotations path for test split
    classes_path: str = "data/classes.txt"              # optional newline-separated class names file
    train_split: str | None = None                      # optional explicit split override (e.g. "train")
    val_split: str | None = None                        # optional explicit split override (e.g. "valid")
    train_augment: bool = True                          # apply train-only photometric augmentation

    # lightweight DETR settings (for real detection training)
    det_num_queries: int = 50
    det_decoder_layers: int = 2
    det_num_heads: int = 8
    det_ff_dim: int = 512
    det_dropout: float = 0.1
    det_cls_loss_coef: float = 1.0
    det_bbox_loss_coef: float = 5.0
    det_giou_loss_coef: float = 2.0
    det_eos_coef: float = 0.1

    # optional bridge from CLIP pretraining to detection
    clip_init_checkpoint: str | None = None            # e.g. "clip_backbone.pth" or checkpoints/...pth
    freeze_backbone_epochs: int = 1                    # freeze image encoder for first N detection epochs

    # evaluation / experiment logging
    eval_conf_thres: float = 0.001                     # low threshold for mAP eval
    eval_top_k: int = 100
    eval_nms_iou: float = 0.0
    experiments_root: str = "experiments"
    class_stats_max_samples: int | None = None
