import json
import tempfile
import unittest
from pathlib import Path


class TestRunBaselineBenchmarks(unittest.TestCase):
    def test_parse_clipter_eval_json(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import run_baseline_benchmarks as rbb  # type: ignore

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "eval_clipter.json"
            payload = {
                "metrics": {
                    "map": 0.123,
                    "map50": 0.234,
                    "map75": 0.111,
                }
            }
            p.write_text(json.dumps(payload), encoding="utf-8")
            map50, map5095, map75, latency, status = rbb._parse_metrics(p)
            self.assertEqual(status, "ok")
            self.assertAlmostEqual(float(map50), 0.234, places=6)
            self.assertAlmostEqual(float(map5095), 0.123, places=6)
            self.assertAlmostEqual(float(map75), 0.111, places=6)
            self.assertIsNone(latency)

    def test_parse_log_like_metrics(self):
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "scripts"))
        import run_baseline_benchmarks as rbb  # type: ignore

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "result.log"
            p.write_text(
                "{ 'PascalBoxes_Precision/mAP@0.5IOU': np.float64(0.22), "
                "'PascalBoxes_Precision/mAP@0.5:0.95IOU': np.float64(0.11) }",
                encoding="utf-8",
            )
            map50, map5095, map75, latency, status = rbb._parse_metrics(p)
            self.assertEqual(status, "ok")
            self.assertAlmostEqual(float(map50), 0.22, places=6)
            self.assertAlmostEqual(float(map5095), 0.11, places=6)
            self.assertIsNone(map75)
            self.assertIsNone(latency)


if __name__ == "__main__":
    unittest.main()
