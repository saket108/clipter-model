"""Create a stratified validation split from YOLO-style `stratified_train`.
- Copies `val_ratio` of images (stratified by primary class) into `stratified_val_10pct/images` and corresponding `labels` by default.
- Safe: copies files, does not delete originals, and avoids overwriting existing `stratified_val`.

Usage:
    python -m clipdetr.utils.create_val_split --root data/weak-2/final_dataset --val_ratio 0.10

Notes:
- Default target split is `stratified_val_10pct` to preserve any existing `stratified_val` folder.
- Use `--val_split_name` to change the destination or `--overwrite` to replace an existing target.
"""
import argparse
from pathlib import Path
import random
import shutil
from collections import defaultdict


def create_stratified_val(root: Path, split_name="stratified_train", val_split_name="stratified_val_10pct", val_ratio=0.1, seed=42, overwrite=False):
    random.seed(seed)
    root = Path(root)
    train_images = root / split_name / "images"
    train_labels = root / split_name / "labels"

    val_root = root / val_split_name
    # safety: if target exists and is non-empty and overwrite is False, abort early
    if val_root.exists() and any(val_root.iterdir()) and not overwrite:
        print(f"Target '{val_root}' already exists and is not empty. Use --overwrite or pass a different --val_split_name.")
        return

    # if overwrite requested, remove existing target so we create a clean val split
    if overwrite and val_root.exists():
        shutil.rmtree(val_root)

    val_images = val_root / "images"
    val_labels = val_root / "labels"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    # collect files by primary class (first class in label file)
    files_by_class = defaultdict(list)
    image_files = sorted([p for p in train_images.glob("**/*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])

    for img in image_files:
        lbl = train_labels / (img.stem + '.txt')
        primary = 'none'
        if lbl.exists():
            with open(lbl, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        primary = parts[0]
                        break
        files_by_class[primary].append(img)

    # sample per-class
    val_files = set()
    for cls, files in files_by_class.items():
        k = max(1, int(len(files) * val_ratio))
        sampled = random.sample(files, k)
        for s in sampled:
            val_files.add(s)

    # copy sampled files
    for img in val_files:
        src_img = img
        src_lbl = train_labels / (img.stem + '.txt')
        dst_img = val_images / img.name
        dst_lbl = val_labels / (img.stem + '.txt')
        shutil.copy2(src_img, dst_img)
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

    print(f"Created val split: {len(val_files)} files copied to {val_images} and {val_labels}")
    print(f"Train size (unchanged): {len(image_files)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--val_split_name', type=str, default='stratified_val_10pct',
                   help='destination split name (default: stratified_val_10pct)')
    p.add_argument('--overwrite', action='store_true', help='overwrite target split if it exists')
    args = p.parse_args()
    create_stratified_val(Path(args.root), val_ratio=args.val_ratio, val_split_name=args.val_split_name, overwrite=args.overwrite)