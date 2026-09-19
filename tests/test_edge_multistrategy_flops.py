"""CPU checks for the declared logical matmul FLOPs boundary."""

import unittest

from tools.summarize_edge_multistrategy import NS, SPATIAL, B, N, P, Q, cached, full, und


class LogicalFlopsTest(unittest.TestCase):
    def test_layout_and_profile(self):
        self.assertEqual(N, 9 * SPATIAL + 33)
        self.assertEqual(NS, P + 8 * 184)
        self.assertEqual(B * 2 * 32 * SPATIAL * Q, 1_247_805_440)

    def test_dense_and_cache_amortization(self):
        prefill = B * (und(158) + und(19))
        pair = B * (full(N, 158) + full(N, 19))
        dense, hit = prefill + 4 * pair, prefill + 2 * pair
        self.assertEqual(dense, 88_301_739_868_160)
        self.assertEqual((dense + hit) / 2, prefill + 3 * pair)
        self.assertGreater((dense + hit) / 2 / dense, 0.75)

    def test_sparse_b6(self):
        total = B * (und(158) + und(19))
        total += B * (full(N, 158) + 3 * full(NS, 158) + 4 * full(NS, 19))
        total += B * 2 * 32 * SPATIAL * Q
        self.assertEqual(total, 53_627_590_180_864)

    def test_toca_refresh_is_mlp_only(self):
        # All future MLP rows refreshed still does not recompute future Q/O/AV.
        self.assertLess(cached(158, 2720), full(N, 158))
        fresh = [int(0.25 * (1.5 - block / 27) * 2720) for block in range(B)]
        self.assertEqual((fresh[0], fresh[-1]), (1020, 340))


if __name__ == "__main__":
    unittest.main()
