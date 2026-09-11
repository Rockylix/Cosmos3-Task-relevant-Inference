"""GPU verification of the shared-QK ToCa kernel against FP32 and native AV."""

import pytest
import torch

from cosmos_framework.inference.toca_future import incoming_future_score


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel test")
@pytest.mark.parametrize("nq,nu,hq,hk,d", [(37, 7, 4, 2, 32), (129, 0, 8, 2, 64), (3093, 158, 16, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_joint_score_and_av(nq, nu, hq, hk, d, dtype):
    from cosmos_framework.inference.toca_joint_attention import joint_attention_score
    from cosmos_framework.model.attention import attention

    torch.manual_seed(72)
    q = torch.randn(nq, hq, d, device="cuda", dtype=dtype)
    ku, kg, vu, vg = [torch.randn(n, hk, d, device="cuda", dtype=dtype) for n in (nu, nq, nu, nq)]
    future = torch.arange(3, nq, 2, device="cuda")
    with torch.inference_mode():
        actual, score = joint_attention_score(q, ku, kg, vu, vg, future, d**-0.5)
        expected_score = incoming_future_score(q, ku, kg, future, d**-0.5)
        expected = attention(
            query=q[None], key=torch.cat((ku, kg))[None], value=torch.cat((vu, vg))[None], is_causal=False
        )[0]
    torch.testing.assert_close(score, expected_score, rtol=1e-4, atol=2e-7)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative < (0.004 if dtype == torch.bfloat16 else 0.001), relative
    assert torch.isfinite(actual).all() and torch.isfinite(score).all()
    # Partial/non-contiguous key columns must retain the full-key denominator.
    assert score.sum() < 1


def test_official_column_score_axis():
    # Official FLUX: attn_map.mean(dim=1).mean(dim=1) on [B,H,Q,K].
    torch.manual_seed(9)
    q, k = torch.randn(1, 4, 9, 8), torch.randn(1, 4, 13, 8)
    probability = (q @ k.transpose(-1, -2) * 8**-0.5).softmax(-1)
    official = probability.mean(dim=1).mean(dim=1)
    assert official.shape == (1, 13)
    torch.testing.assert_close(official.sum(-1), torch.ones(1))
    q_gen = q[0].transpose(0, 1)
    keys = k[0].transpose(0, 1)
    idx = torch.tensor([0, 3, 8])
    actual = incoming_future_score(q_gen, keys[:4], keys[4:], idx, 8**-0.5)
    torch.testing.assert_close(actual, official[0, idx + 4])
