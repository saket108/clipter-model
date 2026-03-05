"""Split a YOLO dataset into custom train/val/test ratios.

Supports two common input layouts:
1) Flat:
   <root>/images, <root>/labels
2) Split folders:
   <root>/<split>/images, <root>/<split>/labels

Examples:
  python -m clipdetr.utils.split_yolo_dataset ^
    --root "C:\\path\\to\\merged_dataset" ^
    --ratios 80,10,10 ^
    --split-names train,valid,test ^
    --overwrite

  python -m clipdetr.utils.split_yolo_dataset ^
    --root "C:\\path\\to\\merged_dataset" ^
    --source-splits train,valid,test ^
    --ratios 70,20,10 ^
    --seed 123 ^
    --overwrite
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_path: Path
    label_path: Optional[Path]
    source_split: str
    primary_class: str
    stem: str
    suffix: str


def _parse_csv(raw: str) -> List[str]:
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if len(out) == 0:
        raise ValueError("Expected a non-empty comma-separated value list.")
    return out


def _parse_ratios(raw: str) -> List[float]:
    vals = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if len(vals) < 2:
        raise ValueError("Provide at least 2 ratio values, e.g. 80,10,10")
    if any(v < 0 for v in vals):
        raise ValueError("Ratios must be non-negative.")
    total = sum(vals)
    if total <= 0:
        raise ValueError("At least one ratio must be > 0.")
    return [v / total for v in vals]


def _read_primary_class(label_path: Optional[Path]) -> str:
    if label_path is None or not label_path.exists():
        return "none"
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) > 0:
                return parts[0]
    except Exception:
        return "none"
    return "none"


def _discover_source_splits(root: Path, source_splits: Optional[str]) -> List[str]:
    if source_splits:
        splits = _parse_csv(source_splits)
        for split in splits:
            if split == ".":
                images_dir = root / "images"
            else:
                images_dir = root / split / "images"
            if not images_dir.exists():
                raise FileNotFoundError(f"Source split '{split}' has no images dir: {images_dir}")
        return splits

    flat_images = root / "images"
    if flat_images.exists():
        return ["."]

    found = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "images").exists():
            found.append(d.name)
    if len(found) == 0:
        raise FileNotFoundError(
            f"Could not detect source layout under: {root}. "
            "Expected either <root>/images or <root>/<split>/images."
        )
    return found


def _collect_samples(root: Path, source_splits: List[str], include_unlabeled: bool) -> List[Sample]:
    samples: List[Sample] = []
    seen_images = set()

    for split in source_splits:
        if split == ".":
            images_dir = root / "images"
            labels_dir = root / "labels"
        else:
            images_dir = root / split / "images"
            labels_dir = root / split / "labels"

        image_files = sorted(
            p for p in images_dir.glob("**/*") if p.is_file() and p.suffix.lower() in IMG_EXTS
        )

        for img in image_files:
            img_key = str(img.resolve())
            if img_key in seen_images:
                continue
            seen_images.add(img_key)

            label_path = labels_dir / f"{img.stem}.txt"
            if not label_path.exists():
                if not include_unlabeled:
                    continue
                label_path = None

            primary = _read_primary_class(label_path)
            samples.append(
                Sample(
                    image_path=img,
                    label_path=label_path,
                    source_split=split,
                    primary_class=primary,
                    stem=img.stem,
                    suffix=img.suffix.lower(),
                )
            )

    if len(samples) == 0:
        raise RuntimeError("No samples were found from the selected source splits.")
    return samples


def _allocate_counts(n: int, ratios: List[float]) -> List[int]:
    raw = [r * n for r in ratios]
    counts = [int(math.floor(x)) for x in raw]
    rem = n - sum(counts)
    if rem > 0:
        order = sorted(range(len(ratios)), key=lambda i: (raw[i] - counts[i]), reverse=True)
        for i in order[:rem]:
            counts[i] += 1
    return counts


def _split_samples(
    samples: List[Sample],
    ratios: List[float],
    split_names: List[str],
    seed: int,
    stratify: bool,
) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    out: Dict[str, List[Sample]] = {name: [] for name in split_names}

    if stratify:
        groups: Dict[str, List[Sample]] = defaultdict(list)
        for s in samples:
            groups[s.primary_class].append(s)
        for _, items in groups.items():
            rng.shuffle(items)
            counts = _allocate_counts(len(items), ratios)
            start = 0
            for i, n in enumerate(counts):
                out[split_names[i]].extend(items[start : start + n])
                start += n
    else:
        pool = samples[:]
        rng.shuffle(pool)
        counts = _allocate_counts(len(pool), ratios)
        start = 0
        for i, n in enumerate(counts):
            out[split_names[i]].extend(pool[start : start + n])
            start += n

    for name in split_names:
        rng.shuffle(out[name])
    return out


def _prepare_output_root(root: Path, output_root: Path, split_names: List[str], overwrite: bool) -> Path:
    # If writing back into the same root, stage in a temp folder first so source data stays intact.
    if output_root.resolve() == root.resolve():
        if not overwrite:
            for split in split_names:
                p = output_root / split
                if p.exists() and any(p.iterdir()):
                    raise FileExistsError(
                        f"Target split folder already exists: {p}. Use --overwrite to replace."
                    )
        stage_root = output_root / f".split_stage_{int(time.time())}"
        if stage_root.exists():
            shutil.rmtree(stage_root)
        stage_root.mkdir(parents=True, exist_ok=False)
        return stage_root

    if overwrite and output_root.exists():
        for split in split_names:
            p = output_root / split
            if p.exists():
                shutil.rmtree(p)

    for split in split_names:
        p = output_root / split
        if p.exists() and any(p.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Target split folder is not empty: {p}. Use --overwrite to replace."
            )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _copy_split_content(split_to_samples: Dict[str, List[Sample]], target_root: Path) -> Dict[str, int]:
    stats = {}
    for split_name, items in split_to_samples.items():
        images_out = target_root / split_name / "images"
        labels_out = target_root / split_name / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        used_stems = set()
        count = 0
        for idx, sample in enumerate(items):
            base = sample.stem
            stem = base
            suffix_id = 1
            while stem in used_stems or (images_out / f"{stem}{sample.suffix}").exists():
                stem = f"{base}_{suffix_id}"
                suffix_id += 1
            used_stems.add(stem)

            dst_img = images_out / f"{stem}{sample.suffix}"
            shutil.copy2(sample.image_path, dst_img)

            if sample.label_path is not None and sample.label_path.exists():
                dst_lbl = labels_out / f"{stem}.txt"
                shutil.copy2(sample.label_path, dst_lbl)
            count = idx + 1

        stats[split_name] = count
    return stats


def _finalize_same_root_write(
    root: Path,
    stage_root: Path,
    split_names: List[str],
    overwrite: bool,
):
    for split in split_names:
        dst = root / split
        src = stage_root / split
        if dst.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Target split folder already exists: {dst}. Use --overwrite to replace."
                )
            shutil.rmtree(dst)
        if src.exists():
            shutil.move(str(src), str(dst))
    shutil.rmtree(stage_root, ignore_errors=True)


def _load_class_names_from_yaml(root: Path):
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None

    names = data.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        def _to_int(k):
            try:
                return int(k)
            except Exception:
                return 10**9

        return [str(v) for _, v in sorted(names.items(), key=lambda kv: _to_int(kv[0]))]
    return None


def _infer_num_classes_from_labels(samples: List[Sample]) -> int:
    max_id = -1
    for sample in samples:
        if sample.label_path is None or not sample.label_path.exists():
            continue
        try:
            for line in sample.label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                cid = int(float(parts[0]))
                if cid > max_id:
                    max_id = cid
        except Exception:
            continue
    return max_id + 1 if max_id >= 0 else 0


def _write_data_yaml(output_root: Path, split_names: List[str], all_samples: List[Sample]):
    train_name = split_names[0]
    val_name = split_names[1]
    test_name = split_names[2] if len(split_names) > 2 else split_names[1]

    names = _load_class_names_from_yaml(output_root)
    if names is None:
        inferred_nc = _infer_num_classes_from_labels(all_samples)
        names = [f"class_{i}" for i in range(inferred_nc)]
    nc = len(names)

    content = {
        "path": str(output_root.resolve()),
        "train": f"{train_name}/images",
        "val": f"{val_name}/images",
        "test": f"{test_name}/images",
        "nc": nc,
        "names": {i: n for i, n in enumerate(names)},
    }

    yaml_path = output_root / "data.yaml"
    try:
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)
    except Exception:
        # Minimal fallback writer to avoid hard dependency at runtime.
        lines = [
            f"path: {content['path']}",
            f"train: {content['train']}",
            f"val: {content['val']}",
            f"test: {content['test']}",
            f"nc: {content['nc']}",
            "names:",
        ]
        for i, name in content["names"].items():
            lines.append(f"  {i}: {name}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="dataset root")
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="destination root (default: same as --root, using safe staging)",
    )
    p.add_argument(
        "--source-splits",
        type=str,
        default=None,
        help="comma-separated source splits to merge before splitting (e.g. train,valid,test). "
             "Default: auto-detect.",
    )
    p.add_argument(
        "--split-names",
        type=str,
        default="train,valid,test",
        help="output split names (must match ratio count)",
    )
    p.add_argument(
        "--ratios",
        type=str,
        default="80,10,10",
        help="split ratios as comma values (percent or fractions), e.g. 80,10,10 or 0.8,0.1,0.1",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="replace existing target split folders")
    p.add_argument("--no-stratify", action="store_true", help="disable stratified split by primary class")
    p.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="include images even when matching .txt label is missing",
    )
    p.add_argument("--no-write-yaml", action="store_true", help="do not write/update output data.yaml")
    p.add_argument("--dry-run", action="store_true", help="print planned counts only")
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    output_root = Path(args.output_root) if args.output_root else root
    split_names = _parse_csv(args.split_names)
    ratios = _parse_ratios(args.ratios)
    if len(split_names) != len(ratios):
        raise ValueError(
            f"split-names count ({len(split_names)}) must match ratios count ({len(ratios)})."
        )

    source_splits = _discover_source_splits(root, args.source_splits)
    print(f"Source splits: {source_splits}")
    samples = _collect_samples(root, source_splits, include_unlabeled=args.include_unlabeled)
    print(f"Collected samples: {len(samples)}")

    class_counts = Counter([s.primary_class for s in samples])
    print(f"Primary-class buckets: {len(class_counts)}")

    split_to_samples = _split_samples(
        samples=samples,
        ratios=ratios,
        split_names=split_names,
        seed=args.seed,
        stratify=(not args.no_stratify),
    )

    print("Planned split sizes:")
    for name in split_names:
        print(f"  {name}: {len(split_to_samples[name])}")

    if args.dry_run:
        return

    target_root = _prepare_output_root(root, output_root, split_names, overwrite=args.overwrite)
    stats = _copy_split_content(split_to_samples, target_root)

    if output_root.resolve() == root.resolve():
        _finalize_same_root_write(
            root=output_root,
            stage_root=target_root,
            split_names=split_names,
            overwrite=args.overwrite,
        )

    if not args.no_write_yaml:
        _write_data_yaml(output_root, split_names, samples)
        print(f"Wrote data.yaml: {output_root / 'data.yaml'}")

    print("Split complete:")
    for name in split_names:
        print(f"  {name}: {stats.get(name, 0)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
