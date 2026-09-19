"""CPU metadata checks for the actual sparse input packing path."""

import unittest
from types import SimpleNamespace

import torch

from cosmos_framework.inference.specprune_future import SpecPruneFuture


class SparseLayoutTest(unittest.TestCase):
    def test_l0_actions_positions_and_true_shorter_pack(self):
        class Net:
            latent_patch_size = 1
            timestep_scale = 0.001
            num_heads = 4
            head_dim = 2
            num_hidden_layers = 28
            config = SimpleNamespace(joint_attn_implementation="two_way")
            vae2llm = torch.nn.Linear(4, 8)

            def _encode_text(self, p):
                result = torch.zeros(p.sequence_length, 8)
                result[p.text_indexes] = 1
                return result, result.dtype

            def _embed_packed_timesteps(self, t, p):
                return torch.zeros(len(t), 8)

            def _encode_action(self, p, h, dtype):
                h[p.action.sequence_indexes] = 2

        model = SimpleNamespace(net=Net(), tensor_kwargs={"device": "cpu", "dtype": torch.float32})
        adapter = SpecPruneFuture(model)
        adapter.spatial = 4
        original = SimpleNamespace(
            sequence_length=42,
            text_indexes=torch.arange(3),
            text_ids=torch.arange(3),
            position_ids=torch.stack((torch.arange(42) * 7, torch.arange(42) * 11, torch.arange(42) * 13)),
            split_lens=[3, 39],
            sample_lens=[42],
            attn_modes=["causal", "full"],
            is_image_batch=False,
            vision=SimpleNamespace(
                sequence_indexes=torch.arange(3, 39), token_shapes=[(9, 2, 2)], timesteps=torch.zeros(8)
            ),
            action=SimpleNamespace(
                sequence_indexes=torch.arange(39, 42), mse_loss_indexes=torch.arange(40, 42), timesteps=torch.zeros(2)
            ),
        )
        patch_ids = torch.tensor([0, 1, 2, 3, 5, 6, 9, 33])
        pack, _, metadata, original_ids = adapter._view(original, patch_ids, torch.zeros(8, 4), torch.zeros(3, 4), 999)
        from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq

        self.assertEqual(len(get_gen_seq(pack)), 11)
        self.assertEqual(metadata.sequence_length, 14)
        torch.testing.assert_close(metadata.position_ids, original.position_ids[:, original_ids], rtol=0, atol=0)
        self.assertTrue(all(i in original_ids.tolist() for i in (3, 4, 5, 6, 39, 40, 41)))
        self.assertNotIn(7, original_ids.tolist())
        self.assertEqual(original.sequence_length, 42)
        self.assertEqual(original.action.sequence_indexes.tolist(), [39, 40, 41])
        torch.testing.assert_close(metadata.action.tokens[0], torch.zeros(3, 4))


if __name__ == "__main__":
    unittest.main()
