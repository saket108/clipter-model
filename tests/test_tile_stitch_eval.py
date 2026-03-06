import unittest

import torch


class TestTileStitchEval(unittest.TestCase):
    def test_stitch_merges_duplicate_tile_predictions(self):
        from clipdetr.utils.detection_metrics import _stitch_tiled_predictions

        predictions = [
            {
                "boxes": torch.tensor([[40.0 / 64.0, 10.0 / 64.0, 60.0 / 64.0, 30.0 / 64.0]], dtype=torch.float32),
                "scores": torch.tensor([0.95], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            },
            {
                "boxes": torch.tensor([[4.0 / 64.0, 10.0 / 64.0, 24.0 / 64.0, 30.0 / 64.0]], dtype=torch.float32),
                "scores": torch.tensor([0.90], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[40.0 / 64.0, 10.0 / 64.0, 60.0 / 64.0, 30.0 / 64.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            },
            {
                "boxes": torch.tensor([[4.0 / 64.0, 10.0 / 64.0, 24.0 / 64.0, 30.0 / 64.0]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.long),
            },
        ]
        image_keys = [
            "sample__x0_y0_tw64_th64_ow100_oh100.jpg",
            "sample__x36_y0_tw64_th64_ow100_oh100.jpg",
        ]

        stitched = _stitch_tiled_predictions(
            predictions=predictions,
            targets_xyxy=targets,
            image_keys=image_keys,
            nms_iou=0.5,
            gt_dedup_iou=0.5,
        )
        self.assertIsNotNone(stitched)
        stitched_predictions, stitched_targets, meta = stitched
        self.assertEqual(meta["tile_keys"], 2)
        self.assertEqual(meta["stitched_images"], 1)
        self.assertEqual(len(stitched_predictions), 1)
        self.assertEqual(len(stitched_targets), 1)
        self.assertEqual(int(stitched_predictions[0]["boxes"].shape[0]), 1)
        self.assertEqual(int(stitched_targets[0]["boxes"].shape[0]), 1)
        expected = torch.tensor([[0.4, 0.1, 0.6, 0.3]], dtype=torch.float32)
        self.assertTrue(torch.allclose(stitched_predictions[0]["boxes"], expected, atol=1e-4))
        self.assertTrue(torch.allclose(stitched_targets[0]["boxes"], expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
