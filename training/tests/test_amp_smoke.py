"""CPU GradScaler regression tests; no GPU allocation is required."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from amp_smoke import calibrated_optimizer_step, SmokeGradientError
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "AMP scaler runtime checks require the server's Torch environment")
class AmpCalibration(unittest.TestCase):
    def setup_case(self, failures, initial_scale=8., enabled=True):
        import torch
        torch.manual_seed(1)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        original = optimizer.step
        optimizer.step = Mock(wraps=original)
        calls = [0]
        def hook(gradient):
            calls[0] += 1
            return torch.full_like(gradient, float("inf")) if calls[0] <= failures else gradient
        model.weight.register_hook(hook)
        scaler = torch.amp.GradScaler("cpu", init_scale=initial_scale, enabled=enabled)
        values = torch.ones(2, 1)
        return model, optimizer, scaler, lambda: model(values).square().mean()

    def test_overflow_skipped_then_one_finite_step(self):
        model, optimizer, scaler, loss = self.setup_case(failures=1)
        result = calibrated_optimizer_step(model, optimizer, scaler, loss, 1.)
        self.assertEqual(result["skipped_overflow_steps"], 1)
        self.assertEqual(result["scale_used"], 4.)
        self.assertEqual([r["status"] for r in result["attempts"]], ["overflow_step_skipped", "finite_optimizer_step_passed"])
        optimizer.step.assert_called_once()

    def test_persistent_nonfinite_gradients_stop_without_updates(self):
        import torch
        model, optimizer, scaler, loss = self.setup_case(failures=100, initial_scale=2.)
        before = [value.detach().clone() for value in model.parameters()]
        with self.assertRaises(SmokeGradientError):
            calibrated_optimizer_step(model, optimizer, scaler, loss, 1.)
        optimizer.step.assert_not_called()
        for old, current in zip(before, model.parameters()):
            torch.testing.assert_close(old, current)

    def test_non_amp_invalid_gradients_fail_immediately(self):
        model, optimizer, scaler, loss = self.setup_case(failures=1, enabled=False)
        with self.assertRaises(SmokeGradientError):
            calibrated_optimizer_step(model, optimizer, scaler, loss, 1.)
        optimizer.step.assert_not_called()

    def test_nonfinite_loss_is_never_accepted(self):
        model, optimizer, scaler, loss = self.setup_case(failures=0)
        with self.assertRaises(SmokeGradientError):
            calibrated_optimizer_step(model, optimizer, scaler, lambda: loss()*float("nan"), 1.)
        optimizer.step.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
