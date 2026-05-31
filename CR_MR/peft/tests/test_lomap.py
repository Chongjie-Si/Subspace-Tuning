# coding=utf-8
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora import Linear, LoraConfig, LoraModel
from peft.utils.save_and_load import get_peft_model_state_dict


class DummyConfig:
    def to_dict(self):
        return {"model_type": "dummy"}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.proj = nn.Linear(4, 3)

    def forward(self, x):
        return self.proj(x)


class LoMAPTester(unittest.TestCase):
    def test_lomap_linear_training_and_merge(self):
        torch.manual_seed(0)
        x = torch.randn(7, 4)
        layer = Linear(4, 3, r=2, lora_alpha=4, lora_dropout=0.0, bias=True, use_map=True, merge_weights=True)
        layer.reset_map_scalars()

        expected = F.linear(x, layer.weight, layer.bias)
        self.assertTrue(torch.allclose(layer(x), expected, atol=1e-5))

        loss = layer(x).pow(2).mean()
        loss.backward()
        for name in ("map_alpha", "map_beta"):
            grad = getattr(layer, name).grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
        for module in (layer.lora_A, layer.lora_B):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())

        layer.train(False)
        merged = layer(x)
        layer.train(True)
        unmerged = layer(x)
        self.assertTrue(torch.allclose(merged, unmerged, atol=1e-5))

    def test_lomap_model_state_dict(self):
        torch.manual_seed(0)
        x = torch.randn(5, 4)
        base = DummyModel()
        expected = base(x).detach()
        config = LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["proj"],
            use_map=True,
            bias="none",
        )
        model = LoraModel(config, base)

        self.assertIsInstance(model.model.proj, Linear)
        self.assertTrue(model.model.proj.use_map)
        self.assertTrue(torch.allclose(model(x), expected, atol=1e-5))

        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all("lora_" in name or "map_" in name for name in trainable))

        state = get_peft_model_state_dict(model)
        self.assertTrue(any("map_alpha" in key for key in state))
        self.assertTrue(any("map_beta" in key for key in state))


if __name__ == "__main__":
    unittest.main()
