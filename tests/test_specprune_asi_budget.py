import json
from pathlib import Path
import unittest
import torch
from cosmos_framework.inference.specprune_exit_plan import ExitConfig,ObservationPlan


class BudgetTest(unittest.TestCase):
    def test_frozen_budget_and_asi_target(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/specprune_asi_budget.json').read_text())
        self.assertEqual(cfg,dict(local_k=58,global_k=72,min_observation_tokens=184,keep_ratio=.9))
        self.assertAlmostEqual((340+7*184)/(8*340),.5985294117647059)

    def test_no_history_does_not_artificially_fill(self):
        p=ObservationPlan(ExitConfig(local_k=58,global_k=72,min_observation_tokens=184))
        p.begin(torch.ones(340,12),191)
        score=torch.arange(340).float()
        p.local(0,score);p.local(1,score)
        n=int(p.active.sum());self.assertEqual(n,58)
        p.prune(10);self.assertEqual(int(p.active.sum()),n)

    def test_proxy_counts_match_implementation(self):
        p=ObservationPlan(ExitConfig(local_k=58,global_k=72,min_observation_tokens=184))
        p.begin(torch.ones(340,12),191)
        p.global_mask[:170]=True;p.dynamic_mask[170:195]=True
        p.local(0,torch.arange(340).float());p.local(1,torch.arange(340).float().flip(0))
        for b in (10,15,20,25):
            before=int(p.active.sum());expected=min(before,max(int(.9*(before+191)),191+184)-191)
            p.prune(b);self.assertEqual(int(p.active.sum()),expected)
        masks=list(p.masks.values())
        for a,b in zip(masks,masks[1:]):self.assertFalse((b&~a).any())


if __name__=='__main__':unittest.main()
