import unittest


class TestPipelineHelpers(unittest.TestCase):
    def test_checkpoint_resolution_priority(self):
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import pipeline  # type: ignore

        summary = {
            "best_map_checkpoint": "a.pth",
            "best_loss_checkpoint": "b.pth",
            "final_checkpoint": "c.pth",
        }
        self.assertEqual(pipeline._resolve_checkpoint(summary, None), "a.pth")

        summary2 = {
            "best_loss_checkpoint": "b.pth",
            "final_checkpoint": "c.pth",
        }
        self.assertEqual(pipeline._resolve_checkpoint(summary2, None), "b.pth")

        summary3 = {"final_checkpoint": "c.pth"}
        self.assertEqual(pipeline._resolve_checkpoint(summary3, None), "c.pth")

    def test_checkpoint_resolution_explicit_override(self):
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import pipeline  # type: ignore

        summary = {"best_map_checkpoint": "a.pth"}
        self.assertEqual(
            pipeline._resolve_checkpoint(summary, "manual.pth"),
            "manual.pth",
        )

    def test_tile_stitch_resolution_defaults_to_tiled_runs(self):
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import pipeline  # type: ignore

        self.assertTrue(pipeline._resolve_tile_stitch_eval(None, 640))
        self.assertFalse(pipeline._resolve_tile_stitch_eval(None, 0))
        self.assertFalse(pipeline._resolve_tile_stitch_eval(False, 640))
        self.assertTrue(pipeline._resolve_tile_stitch_eval(True, 0))


if __name__ == "__main__":
    unittest.main()
