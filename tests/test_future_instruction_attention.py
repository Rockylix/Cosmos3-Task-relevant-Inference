"""CPU utilities tests, not model/kernel or RoboLab validation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.inference.future_instruction_attention import (
    capture_future_instruction_rows,
    compare_attention_output,
    export_heatmaps,
    instruction_attention_mass,
    scatter_future_grid,
)


class InstructionAttentionTest(unittest.TestCase):
    def test_callback_adapter_uses_all_ar_and_gen_keys(self):
        arguments = dict(
            q_gen=torch.zeros(5, 4, 8),
            k_ar=torch.zeros(3, 2, 8),
            k_gen=torch.zeros(5, 2, 8),
            v_ar=torch.ones(3, 2, 8),
            v_gen=torch.ones(5, 2, 8),
            attn_output_gen=torch.ones(5, 4, 8),
            scaling=8**-0.5,
            future_query_indices=torch.tensor([1, 3]),
            instruction_ar_indices=torch.tensor([1]),
            full_gen_attention_verified=True,
        )
        result, metrics = capture_future_instruction_rows(**arguments)
        torch.testing.assert_close(result.mass, torch.full((2,), 1 / 8))
        self.assertEqual(metrics["relative_l2"], 0)
        with self.assertRaisesRegex(ValueError, "Full GEN"):
            capture_future_instruction_rows(**(arguments | {"full_gen_attention_verified": False}))
        with self.assertRaisesRegex(ValueError, "AR text"):
            capture_future_instruction_rows(**(arguments | {"instruction_ar_indices": torch.tensor([3])}))

    def test_mass_denominator_includes_non_instruction_keys(self):
        q = torch.zeros(3, 4, 8)
        k = torch.zeros(10, 2, 8)
        result = instruction_attention_mass(q, k, torch.ones_like(k), torch.tensor([1, 3]), scale=8**-0.5)
        torch.testing.assert_close(result.mass, torch.full((3,), 0.2))
        torch.testing.assert_close(result.output, torch.ones_like(q))

    def test_gqa_output_matches_sdpa(self):
        generator = torch.Generator().manual_seed(10)
        q = torch.randn(7, 16, 8, generator=generator)
        k = torch.randn(19, 8, 8, generator=generator)
        v = torch.randn(19, 8, 8, generator=generator)
        result = instruction_attention_mass(q, k, v, torch.tensor([0, 2, 4]), scale=8**-0.5, query_batch_size=3)
        reference = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            v.transpose(0, 1)[None],
            enable_gqa=True,
        )[0].transpose(0, 1)
        torch.testing.assert_close(result.output, reference, atol=1e-6, rtol=1e-5)
        self.assertLess(compare_attention_output(result.output, reference)["relative_l2"], 1e-6)

    def test_visible_mask_and_batching(self):
        q = torch.zeros(3, 4, 8)
        k = torch.zeros(10, 2, 8)
        visible = torch.zeros(3, 10, dtype=torch.bool)
        visible[:, :5] = True
        result = instruction_attention_mass(
            q, k, k, torch.tensor([0, 7]), scale=1, visible_mask=visible, query_batch_size=1
        )
        torch.testing.assert_close(result.mass, torch.full((3,), 0.2))
        visible[1] = False
        with self.assertRaisesRegex(ValueError, "no visible keys"):
            instruction_attention_mass(q, k, k, torch.tensor([0]), scale=1, visible_mask=visible)

    def test_invalid_scores_and_instruction_span(self):
        q = torch.zeros(3, 4, 8)
        k = torch.zeros(10, 2, 8)
        for ids in (torch.tensor([], dtype=torch.long), torch.tensor([1, 1]), torch.tensor([10])):
            with self.assertRaises(ValueError):
                instruction_attention_mass(q, k, k, ids, scale=1)
        q[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            instruction_attention_mass(q, k, k, torch.tensor([0]), scale=1)

    def test_original_coordinate_scatter_and_missing_not_zero(self):
        grid = scatter_future_grid(torch.tensor([0.1, 0.4]), torch.tensor([5, 1]), frames=2, height=2, width=2)
        self.assertAlmostEqual(float(grid[1, 0, 1]), 0.1)
        self.assertAlmostEqual(float(grid[0, 0, 1]), 0.4)
        self.assertTrue(np.isnan(grid[0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            scatter_future_grid(torch.tensor([0.1, 0.2]), torch.tensor([1, 1]), frames=2, height=2, width=2)

    def test_export_shared_scale_and_raw_data(self):
        grids = {
            "synthetic_B0": np.full((8, 2, 3), 0.1, dtype=np.float32),
            "synthetic_B1": np.full((8, 2, 3), 0.5, dtype=np.float32),
        }
        grids["synthetic_B0"][0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory(prefix="instruction-heatmap-test-") as temp:
            paths = export_heatmaps(grids, Path(temp), metadata={"synthetic_test_only": True})
            self.assertTrue(all(path.exists() for path in paths))
            details = json.loads((Path(temp) / "metadata.json").read_text())
            self.assertEqual(details["color_vmax"], 0.5)
            with np.load(Path(temp) / "scores.npz") as data:
                np.testing.assert_equal(data["synthetic_B0"], grids["synthetic_B0"])
            with self.assertRaises(FileExistsError):
                export_heatmaps(grids, Path(temp), metadata={})


if __name__ == "__main__":
    unittest.main()
