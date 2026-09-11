import ast
import math
import unittest
from pathlib import Path

import torch


def load_metrics():
    source = Path(__file__).resolve().parents[1] / "cosmos_framework/scripts/worldcache_smoke.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "metrics")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["metrics"]


class MetricsTests(unittest.TestCase):
    def test_long_rgb_vector(self):
        metrics = load_metrics()
        y = (torch.arange(2_000_000) % 256).float() / 255
        self.assertAlmostEqual(metrics(y, y)["cosine"], 1.0, places=12)
        x = y.clone()
        x[::17] = 0
        m = metrics(x, y)
        xd, yd = x.double(), y.double()
        expected = float((xd * yd).sum() / (xd.square().sum() * yd.square().sum()).sqrt())
        self.assertAlmostEqual(m["cosine"], expected, places=12)
        self.assertLessEqual(m["cosine"], 1.0)

    def test_small_vector(self):
        m = load_metrics()(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 1.0]))
        self.assertEqual(m["mse"], 1.0)
        self.assertAlmostEqual(m["relative_l2"], 1.0)
        self.assertAlmostEqual(m["cosine"], 1 / math.sqrt(2))

    def test_invalid(self):
        with self.assertRaises(FloatingPointError):
            load_metrics()(torch.tensor([float("nan")]), torch.ones(1))
        with self.assertRaises(ValueError):
            load_metrics()(torch.ones(2), torch.ones(1))


if __name__ == "__main__":
    unittest.main()
