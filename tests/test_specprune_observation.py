import unittest

import torch

from cosmos_framework.inference.specprune_future import ChunkSelector, SpecPruneConfig
from cosmos_framework.inference.specprune_observation import condition_rgb, observation_patches


class ObservationTest(unittest.TestCase):
    def patches(self, rgb):
        return observation_patches(
            rgb, latent_hw=(33, 40), latent_patch_size=2, spatial_factor=16, image_size=[544, 736, 540, 640]
        )

    def test_native_floor_crop_and_patch_padding(self):
        rgb = torch.ones(3, 544, 736, dtype=torch.uint8)
        x, meta = self.patches(rgb)
        self.assertEqual(x.shape, (340, 3 * 32 * 32))
        self.assertEqual(meta["compared_rgb_hw"], [528, 640])
        self.assertTrue((x[-1].reshape(3, 32, 32)[:, :16] == 1).all())
        self.assertTrue((x[-1].reshape(3, 32, 32)[:, 16:] == 0).all())
        rgb[:, 528:] = 200
        rgb[:, :, 640:] = 200
        self.assertTrue(torch.equal(self.patches(rgb)[0], x))

    def test_spatial_index_preserved(self):
        rgb = torch.ones(3, 544, 736, dtype=torch.uint8)
        x = self.patches(rgb)[0]
        rgb[:, 2 * 32 + 5, 3 * 32 + 7] = 100
        changed = (self.patches(rgb)[0] != x).any(-1)
        self.assertEqual(torch.where(changed)[0].tolist(), [2 * 20 + 3])

    def test_extract_only_condition_and_copy(self):
        video = torch.zeros(3, 33, 544, 736, dtype=torch.uint8)
        video[:, 0] = 25
        video[:, 1:] = 255
        rgb = condition_rgb({"video": [[video]]})
        video.zero_()
        self.assertTrue((rgb == 25).all())
        with self.assertRaises(ValueError):
            condition_rgb({"video": [[video.float()]]})

    def test_refuse_guessed_grid(self):
        with self.assertRaisesRegex(ValueError, "crop mismatch"):
            observation_patches(
                torch.ones(3, 544, 736, dtype=torch.uint8),
                latent_hw=(34, 40),
                latent_patch_size=2,
                spatial_factor=16,
                image_size=[544, 736, 540, 640],
            )

    def selector_history(self, observation):
        selector = ChunkSelector(SpecPruneConfig(dynamic_source="observation"))
        scores = torch.arange(340).float().expand(8, -1)
        selector.begin(torch.randn(8, 340, 16), observation=observation)
        _, info = selector.choose(scores, -scores)
        self.assertEqual(info["dynamic_counts"], [0] * 8)
        history = scores.masked_fill(~selector.selected, torch.nan)
        selector.complete({13: history, 27: history})
        return selector, scores

    def test_identical_observation_independent_noise(self):
        observation = torch.ones(340, 12)
        selector, scores = self.selector_history(observation)
        selector.begin(torch.randn(8, 340, 16) * 100, observation=observation)
        mask, info = selector.choose(scores, scores)
        self.assertEqual(info["dynamic_counts"], [27] * 8)
        self.assertLess(int(mask.sum()), 2720)
        self.assertTrue(info["dynamic_shared_across_frames"])
        self.assertIsNone(info["initial_noise_similarity_mean"])
        self.assertAlmostEqual(info["observation_similarity_mean"], 1.0, places=6)

    def test_changed_patch_protected_in_every_frame_and_reset(self):
        observation = torch.ones(340, 12)
        selector, scores = self.selector_history(observation)
        observation[123, :6] = 0
        selector.begin(torch.randn(8, 340, 16), observation=observation)
        mask, info = selector.choose(scores, scores)
        self.assertTrue(selector.last_dynamic_mask[:, 123].all())
        self.assertTrue(mask[:, 123].all())
        self.assertTrue(info["dynamic_shared_across_frames"])
        selector.reset()
        self.assertIsNone(selector.previous_observation)
        self.assertIsNone(selector.pending_observation)

    def test_missing_observation_rejected(self):
        selector = ChunkSelector(SpecPruneConfig(dynamic_source="observation"))
        with self.assertRaisesRegex(ValueError, "observation"):
            selector.begin(torch.randn(8, 340, 16))


if __name__ == "__main__":
    unittest.main()
