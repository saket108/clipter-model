#!/usr/bin/env python3
"""Verify that the CLIPTER runtime environment is usable."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_version(name: str) -> str:
    mod = importlib.import_module(name)
    return str(getattr(mod, "__version__", "unknown"))


def _verify_detector() -> dict[str, object]:
    root = _repo_root()
    clipdetr_root = root / "clipdetr"
    for path in (root, clipdetr_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from models.light_detr import LightDETR
    from losses.detection_loss import DetectionLoss, SCIPY_AVAILABLE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightDETR(
        num_classes=5,
        hidden_dim=288,
        num_queries=10,
        decoder_layers=1,
        num_heads=4,
        ff_dim=384,
        dropout=0.0,
        image_backbone="mobilenet_v3_small",
        image_pretrained=False,
        use_multiscale_memory=True,
        multiscale_levels=3,
    ).to(device)
    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        outputs = model(x)
    assert outputs["pred_logits"].shape == (2, 10, 6)
    assert outputs["pred_boxes"].shape == (2, 10, 4)

    criterion = DetectionLoss(num_classes=5).to(device)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=device),
            "labels": torch.tensor([1], dtype=torch.long, device=device),
        },
        {
            "boxes": torch.tensor([[0.4, 0.4, 0.1, 0.1]], dtype=torch.float32, device=device),
            "labels": torch.tensor([0], dtype=torch.long, device=device),
        },
    ]
    losses = criterion(outputs, targets)
    loss_total = float(losses["loss_total"].detach().cpu().item())
    return {
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": _module_version("torchvision"),
        "numpy": _module_version("numpy"),
        "Pillow": _module_version("PIL"),
        "PyYAML": _module_version("yaml"),
        "tqdm": _module_version("tqdm"),
        "scipy": _module_version("scipy"),
        "scipy_hungarian_available": bool(SCIPY_AVAILABLE),
        "detector_forward_ok": True,
        "sample_loss_total": loss_total,
    }


def _verify_text_path() -> dict[str, object]:
    root = _repo_root()
    clipdetr_root = root / "clipdetr"
    for path in (root, clipdetr_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from tokenizer import SimpleTokenizer

    tok = SimpleTokenizer(max_length=8)
    ids = tok.encode(["surface crack near panel seam"])
    return {
        "transformers": _module_version("transformers"),
        "tokenizer_ok": True,
        "token_shape": list(ids.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CLIPTER environment.")
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="Also verify the tokenizer/transformers path used by CLIP pretraining.",
    )
    args = parser.parse_args()

    report = {"detector": _verify_detector()}
    if args.with_text:
        report["text"] = _verify_text_path()

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
