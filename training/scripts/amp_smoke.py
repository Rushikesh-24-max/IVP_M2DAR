"""Bounded AMP calibration for a disposable optimizer step, with full diagnostics."""


class SmokeGradientError(ValueError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def calibrated_optimizer_step(model, optimizer, scaler, forward_loss, max_norm, max_attempts=17):
    import torch
    attempts = []
    cpu_rng = torch.get_rng_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
    for attempt in range(1, max_attempts + 1):
        # Reuse the same batch and dropout realization while calibrating scale.
        torch.set_rng_state(cpu_rng)
        for device, state in cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        optimizer.zero_grad(set_to_none=True)
        scale = float(scaler.get_scale())
        loss = forward_loss()
        row = dict(attempt=attempt, scale=scale)
        if not torch.isfinite(loss).item():
            row.update(status="nonfinite_loss")
            attempts.append(row)
            raise SmokeGradientError("Non-finite unscaled smoke-test loss", attempts)
        row["loss"] = float(loss.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad = []
        count = 0
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                count += 1
                if not torch.isfinite(parameter.grad).all().item():
                    bad.append(name)
        row.update(gradient_tensors=count, nonfinite_gradient_tensors=len(bad), first_nonfinite_parameters=bad[:8])
        if not count:
            attempts.append(dict(row, status="no_gradients"))
            raise SmokeGradientError("No parameter gradients were produced", attempts)
        if bad:
            if not scaler.is_enabled() or scale <= 1:
                attempts.append(dict(row, status="persistent_nonfinite_gradients"))
                raise SmokeGradientError("Non-finite gradients persist without a large AMP scale; inspect diagnostics", attempts)
            # GradScaler detects these same inf/NaN values and skips optimizer.step.
            scaler.step(optimizer)
            scaler.update()
            next_scale = float(scaler.get_scale())
            row.update(status="overflow_step_skipped", next_scale=next_scale)
            attempts.append(row)
            print(f"AMP calibration {attempt}: scale {scale:g} -> {next_scale:g}; update skipped; first non-finite gradient: {bad[0]}", flush=True)
            if next_scale >= scale:
                raise SmokeGradientError("AMP did not reduce the scale after detecting non-finite gradients", attempts)
            optimizer.zero_grad(set_to_none=True)
            del loss
            continue
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not torch.isfinite(norm).item():
            attempts.append(dict(row, status="nonfinite_global_norm"))
            raise SmokeGradientError("Gradient norm is non-finite despite finite individual gradients", attempts)
        scaler.step(optimizer)
        scaler.update()
        invalid_parameters = [name for name,p in model.named_parameters() if not torch.isfinite(p).all().item()]
        if invalid_parameters:
            attempts.append(dict(row, status="nonfinite_parameters_after_step", first_invalid_parameters=invalid_parameters[:8]))
            raise SmokeGradientError("The optimizer produced non-finite parameters", attempts)
        row.update(status="finite_optimizer_step_passed", gradient_norm=float(norm), next_scale=float(scaler.get_scale()))
        attempts.append(row)
        optimizer.zero_grad(set_to_none=True)
        print(f"Finite gradients and optimizer step passed at AMP scale {scale:g} (attempt {attempt}).", flush=True)
        return dict(loss=row["loss"], gradient_norm=row["gradient_norm"], scale_used=scale,
                    attempts=attempts, skipped_overflow_steps=sum(r["status"]=="overflow_step_skipped" for r in attempts))
    raise SmokeGradientError("AMP calibration exhausted its retry budget; inspect diagnostics", attempts)
