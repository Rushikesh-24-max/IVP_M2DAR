"""CPU numerical tests; full CUDA memory suitability is checked by preflight."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def manual_attention(q, k, v, scale):
    return ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1) @ v


@unittest.skipUnless(HAS_TORCH, "Numerical attention checks require the server's Torch environment")
class AttentionEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from common import activate_source
        activate_source()
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(min(2, cls.old_threads))

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.old_threads)

    def test_non_square_attention_outputs_and_gradients(self):
        import torch
        from slowfast.models.memory_attention import pooled_scaled_attention
        torch.manual_seed(12)
        inputs = [torch.randn(2, 3, size, 16, dtype=torch.float64, requires_grad=True) for size in (17, 7, 7)]
        reference = [value.detach().clone().requires_grad_(True) for value in inputs]
        output = pooled_scaled_attention(*inputs, scale=16**-.5)
        expected = manual_attention(*reference, scale=16**-.5)
        weights = torch.randn_like(output)
        gradients = torch.autograd.grad((output*weights).sum(), inputs)
        reference_gradients = torch.autograd.grad((expected*weights).sum(), reference)
        torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)
        for actual, reference_gradient in zip(gradients, reference_gradients):
            torch.testing.assert_close(actual, reference_gradient, rtol=1e-8, atol=1e-9)

    def test_pooled_module_residual_weights_and_gradients(self):
        import torch
        from slowfast.models.attention import MultiScaleAttention
        torch.manual_seed(42)
        model = MultiScaleAttention(dim=32, num_heads=2, qkv_bias=True, drop_rate=0.0,
                                    kernel_q=(3,3,3), kernel_kv=(3,3,3),
                                    stride_q=(1,1,1), stride_kv=(1,2,2),
                                    has_cls_embed=False, use_query_residual_pool=True).double()
        reference = copy.deepcopy(model)
        inputs = torch.randn(1, 72, 32, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        output, shape = model(inputs, [2,6,6])
        with patch("slowfast.models.attention.pooled_scaled_attention", side_effect=manual_attention):
            expected, expected_shape = reference(reference_inputs, [2,6,6])
        self.assertEqual(shape, expected_shape)
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        torch.testing.assert_close(output, expected, rtol=1e-9, atol=1e-10)
        weights = torch.randn_like(output)
        (output*weights).sum().backward()
        (expected*weights).sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-8, atol=1e-9)
        for (name, value), (reference_name, reference_value) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, reference_name)
            torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
