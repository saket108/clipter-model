import tempfile
import unittest
from pathlib import Path

from PIL import Image


class TestBuildTiledYoloDataset(unittest.TestCase):
    def test_build_tiled_dataset_outputs_tiles_and_yaml(self):
        from clipdetr.utils.build_tiled_yolo_dataset import build_tiled_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            out_dir = Path(tmp) / "tiles"
            for split in ("train", "valid"):
                (root / split / "images").mkdir(parents=True, exist_ok=True)
                (root / split / "labels").mkdir(parents=True, exist_ok=True)
                image_path = root / split / "images" / "sample.jpg"
                Image.new("RGB", (100, 100), color=(255, 255, 255)).save(image_path)
                (root / split / "labels" / "sample.txt").write_text(
                    "0 0.50 0.50 0.30 0.30\n",
                    encoding="utf-8",
                )

            (root / "data.yaml").write_text(
                "train: train/images\nval: valid/images\nnames: ['dent']\n",
                encoding="utf-8",
            )

            report = build_tiled_dataset(
                root=root,
                out_dir=out_dir,
                data_yaml=root / "data.yaml",
                tile_size=64,
                overlap=0.25,
                min_cover=0.2,
                tile_splits=["train", "valid"],
                include_empty_tiles=False,
            )

            self.assertEqual(report["tile_size"], 64)
            self.assertTrue((out_dir / "data.yaml").exists())
            train_tiles = list((out_dir / "train" / "images").glob("*.jpg"))
            valid_tiles = list((out_dir / "valid" / "images").glob("*.jpg"))
            self.assertGreaterEqual(len(train_tiles), 1)
            self.assertGreaterEqual(len(valid_tiles), 1)
            self.assertIn("__x", train_tiles[0].name)
            tiled_label = out_dir / "train" / "labels" / f"{train_tiles[0].stem}.txt"
            self.assertTrue(tiled_label.exists())


if __name__ == "__main__":
    unittest.main()
