"""Build a tiled YOLO dataset for small-object training and stitched eval.

Output tile file names encode original-image geometry:
  <base>__xX_yY_twTW_thTH_owOW_ohOH.jpg
This allows eval-time stitching back into full-image coordinates.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import failure path
        raise RuntimeError("PyYAML is required for tiled dataset generation.") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict in {path}, got {type(data).__name__}")
    return data


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import failure path
        raise RuntimeError("PyYAML is required for tiled dataset generation.") from exc

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _split_name_from_yaml_entry(entry: Any) -> str | None:
    if entry is None:
        return None
    raw = str(entry).strip()
    if not raw:
        return None
    p = Path(raw)
    if p.name.lower() == "images":
        p = p.parent
    return p.name if p.name else None


def _discover_splits(root: Path, data_yaml: Path | None, splits: list[str] | None) -> list[str]:
    if splits:
        return [s for s in splits if (root / s / "images").exists()]

    ordered: list[str] = []
    if data_yaml is not None and data_yaml.exists():
        cfg = _load_yaml(data_yaml)
        for key in ("train", "val", "test"):
            split = _split_name_from_yaml_entry(cfg.get(key))
            if split and split not in ordered:
                ordered.append(split)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "images").exists() and (child / "labels").exists() and child.name not in ordered:
            ordered.append(child.name)
    if not ordered:
        raise FileNotFoundError(f"No YOLO split folders found under {root}")
    return ordered


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = [float(x) for x in parts[1:5]]
        out.append((cls, xc, yc, w, h))
    return out


def _grid_positions(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    positions = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return positions


def _intersection_ratio(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    if area_a <= 0.0:
        return 0.0
    return inter / area_a


def _copy_split(src_split: Path, dst_split: Path) -> dict[str, int]:
    shutil.copytree(src_split / "images", dst_split / "images", dirs_exist_ok=True)
    shutil.copytree(src_split / "labels", dst_split / "labels", dirs_exist_ok=True)
    num_images = sum(1 for _ in (dst_split / "images").rglob("*") if _.is_file())
    num_labels = sum(1 for _ in (dst_split / "labels").rglob("*") if _.is_file())
    return {"mode": "copied", "images": int(num_images), "labels": int(num_labels)}


def _tile_single_image(
    image_path: Path,
    label_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    tile_prefix: str,
    tile_size: int,
    overlap: float,
    min_cover: float,
    include_empty_tiles: bool,
) -> dict[str, int]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size
        labels = _read_yolo_labels(label_path)

        boxes_abs: list[tuple[int, tuple[float, float, float, float]]] = []
        for cls, xc, yc, bw, bh in labels:
            w = bw * float(img_w)
            h = bh * float(img_h)
            cx = xc * float(img_w)
            cy = yc * float(img_h)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            boxes_abs.append((cls, (x1, y1, x2, y2)))

        xs = _grid_positions(img_w, tile_size, overlap)
        ys = _grid_positions(img_h, tile_size, overlap)

        tiles = 0
        tiles_with_boxes = 0
        boxes_written = 0

        for x in xs:
            for y in ys:
                x2 = min(img_w, x + tile_size)
                y2 = min(img_h, y + tile_size)
                tw = int(x2 - x)
                th = int(y2 - y)
                tile_labels: list[str] = []

                for cls, box in boxes_abs:
                    ratio = _intersection_ratio(box, (float(x), float(y), float(x2), float(y2)))
                    if ratio < float(min_cover):
                        continue
                    ix1 = max(box[0], float(x)) - float(x)
                    iy1 = max(box[1], float(y)) - float(y)
                    ix2 = min(box[2], float(x2)) - float(x)
                    iy2 = min(box[3], float(y2)) - float(y)
                    bw = max(0.0, ix2 - ix1)
                    bh = max(0.0, iy2 - iy1)
                    if bw <= 1.0 or bh <= 1.0:
                        continue
                    xc = (ix1 + 0.5 * bw) / float(tw)
                    yc = (iy1 + 0.5 * bh) / float(th)
                    tile_labels.append(f"{cls} {xc:.6f} {yc:.6f} {bw / float(tw):.6f} {bh / float(th):.6f}")

                if not tile_labels and not include_empty_tiles:
                    continue

                tile_name = f"{tile_prefix}__x{x}_y{y}_tw{tw}_th{th}_ow{img_w}_oh{img_h}{image_path.suffix.lower()}"
                out_img_path = out_img_dir / tile_name
                out_lbl_path = out_lbl_dir / f"{Path(tile_name).stem}.txt"

                img.crop((x, y, x2, y2)).save(out_img_path)
                out_lbl_path.write_text(
                    "\n".join(tile_labels) + ("\n" if tile_labels else ""),
                    encoding="utf-8",
                )

                tiles += 1
                if tile_labels:
                    tiles_with_boxes += 1
                    boxes_written += len(tile_labels)

        return {
            "tiles": int(tiles),
            "tiles_with_boxes": int(tiles_with_boxes),
            "boxes_written": int(boxes_written),
        }


def build_tiled_dataset(
    *,
    root: Path,
    out_dir: Path,
    data_yaml: Path | None,
    tile_size: int,
    overlap: float,
    min_cover: float,
    tile_splits: list[str] | None = None,
    include_empty_tiles: bool = False,
) -> dict[str, Any]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("overlap must be in [0, 1)")

    split_names = _discover_splits(root, data_yaml, tile_splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_report: dict[str, Any] = {}

    for split in split_names:
        src_split = root / split
        dst_split = out_dir / split
        (dst_split / "images").mkdir(parents=True, exist_ok=True)
        (dst_split / "labels").mkdir(parents=True, exist_ok=True)

        if tile_splits is not None and split not in tile_splits:
            split_report[split] = _copy_split(src_split, dst_split)
            continue

        totals = {"tiles": 0, "tiles_with_boxes": 0, "boxes_written": 0}
        image_files = sorted(
            [p for p in (src_split / "images").rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )
        for image_path in image_files:
            rel = image_path.relative_to(src_split / "images")
            tile_prefix = rel.with_suffix("").as_posix().replace("/", "__")
            label_path = src_split / "labels" / rel.with_suffix(".txt")
            stats = _tile_single_image(
                image_path=image_path,
                label_path=label_path,
                out_img_dir=dst_split / "images",
                out_lbl_dir=dst_split / "labels",
                tile_prefix=tile_prefix,
                tile_size=tile_size,
                overlap=overlap,
                min_cover=min_cover,
                include_empty_tiles=include_empty_tiles,
            )
            totals["tiles"] += stats["tiles"]
            totals["tiles_with_boxes"] += stats["tiles_with_boxes"]
            totals["boxes_written"] += stats["boxes_written"]

        split_report[split] = {"mode": "tiled", **totals}

    cfg_in = _load_yaml(data_yaml) if data_yaml is not None and data_yaml.exists() else {}
    cfg_out = dict(cfg_in)
    for key in ("train", "val", "test"):
        split = _split_name_from_yaml_entry(cfg_in.get(key)) if isinstance(cfg_in, dict) else None
        if split:
            cfg_out[key] = str((out_dir / split / "images").relative_to(out_dir)).replace("\\", "/")
    if "path" in cfg_out:
        cfg_out["path"] = "."
    _dump_yaml(out_dir / "data.yaml", cfg_out)

    report = {
        "source_root": str(root),
        "output_root": str(out_dir),
        "tile_size": int(tile_size),
        "overlap": float(overlap),
        "min_cover": float(min_cover),
        "tile_splits": list(tile_splits or split_names),
        "include_empty_tiles": bool(include_empty_tiles),
        "splits": split_report,
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Build tiled YOLO dataset for CLIPTER.")
    p.add_argument("--root", required=True, help="Dataset root containing split folders.")
    p.add_argument("--data-yaml", default="data.yaml", help="Path/name of source data.yaml.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--tile-size", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--min-cover", type=float, default=0.35)
    p.add_argument("--tile-splits", default=None, help="Comma-separated split names to tile. Others are copied.")
    p.add_argument("--include-empty-tiles", action="store_true")
    p.add_argument("--report-json", default=None)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_absolute():
        data_yaml = root / data_yaml
    report = build_tiled_dataset(
        root=root,
        out_dir=Path(args.output_root).expanduser().resolve(),
        data_yaml=data_yaml,
        tile_size=int(args.tile_size),
        overlap=float(args.overlap),
        min_cover=float(args.min_cover),
        tile_splits=_parse_csv(args.tile_splits) if args.tile_splits else None,
        include_empty_tiles=bool(args.include_empty_tiles),
    )
    if args.report_json:
        out = Path(args.report_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
        print(f"Tiling report: {out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
