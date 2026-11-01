import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))


class ProbeTests(unittest.TestCase):
    def test_both_architectures(self):
        import torch
        from rq71.models import build_linear_probe
        torch.set_num_threads(1)
        for name in ('mobilenet_v3_small','efficientnet_b0'):
            model=build_linear_probe(name,38,pretrained=False).train()
            values=model(torch.randn(2,3,64,64))
            self.assertEqual(tuple(values.shape),(2,38))
            values.sum().backward()
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
            self.assertIsNotNone(model.head.weight.grad)

    def test_frozen_buffers_and_trainable_head(self):
        import torch
        from torch import nn
        from rq71.models import FrozenLinearProbe
        torch.set_num_threads(1)
        backbone = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(.5))
        model = FrozenLinearProbe(backbone, 4, 2).train()
        before = {k: v.clone() for k, v in backbone.state_dict().items()}
        model(torch.randn(8, 4)).sum().backward()
        self.assertFalse(backbone.training)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in backbone.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
