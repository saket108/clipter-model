import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return torch.ones((len(texts), 32), dtype=torch.long)


class TestStructuredJsonDataset(unittest.TestCase):
    def _write_image(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(128, 128, 128)).save(path)

    def _write_dataset(self, root: Path):
        train_image = root / "train" / "images" / "sample.jpg"
        train_label = root / "train" / "labels" / "sample.txt"
        self._write_image(train_image)
        train_label.parent.mkdir(parents=True, exist_ok=True)
        train_label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

        payload = {
            "dataset_info": {
                "class_definitions": {
                    "crack": "Crack definition",
                    "dent": "Dent definition",
                }
            },
            "images": [
                {
                    "image_id": "sample",
                    "file_name": "sample.jpg",
                    "split": "train",
                    "annotations": [
                        {
                            "annotation_id": "sample_0",
                            "category_id": 0,
                            "category_name": "crack",
                            "zone_estimation": "central",
                            "bounding_box_normalized": {
                                "x_center": 0.4,
                                "y_center": 0.6,
                                "width": 0.2,
                                "height": 0.3,
                            },
                            "risk_assessment": {"severity_level": "moderate"},
                            "description": (
                                "Moderate severity Crack detected in the central structural region. "
                                "Additional boilerplate text."
                            ),
                        }
                    ],
                }
            ],
        }
        (root / "train.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_auto_prefers_yolo_for_detection_when_labels_exist(self):
        from clipdetr.datasets.yolo_dataset import YOLODataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_dataset(root)

            dataset = YOLODataset(
                root=str(root),
                split="train",
                classes=["crack", "dent"],
                tokenizer=None,
                annotation_format="auto",
            )

            self.assertEqual(dataset.annotation_format, "yolo")
            _, _, boxes, class_ids = dataset[0]
            self.assertTrue(torch.allclose(boxes, torch.tensor([[0.5, 0.5, 0.25, 0.25]])))
            self.assertEqual(class_ids.tolist(), [0])

    def test_auto_uses_structured_json_for_text_training(self):
        from clipdetr.datasets.yolo_dataset import YOLODataset

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_dataset(root)
            tokenizer = DummyTokenizer()

            dataset = YOLODataset(
                root=str(root),
                split="train",
                classes=["crack", "dent"],
                tokenizer=tokenizer,
                annotation_format="auto",
                caption_style="auto",
            )

            self.assertEqual(dataset.annotation_format, "structured_json")
            self.assertEqual(dataset.num_classes, 2)
            caption = dataset._compose_caption(dataset.records[0], dataset.records[0]["class_ids"])
            self.assertEqual(
                caption,
                "moderate severity crack detected in the central structural region",
            )

            _, token_ids, boxes, class_ids = dataset[0]
            self.assertEqual(tuple(token_ids.shape), (32,))
            self.assertEqual(class_ids.tolist(), [0])
            self.assertTrue(torch.allclose(boxes, torch.tensor([[0.4, 0.6, 0.2, 0.3]])))
            self.assertEqual(
                tokenizer.calls[0],
                "moderate severity crack detected in the central structural region",
            )


if __name__ == "__main__":
    unittest.main()
