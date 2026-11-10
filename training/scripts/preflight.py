"""Data/config checks and, on the server, a real GPU forward/backward smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
import sys

from common import (application_hash, apply_gpu_memory_limit, load_torch_checkpoint, read_cfg, require_gpu, require_shapes,
                    seed_runtime, sha256, source_integrity, verify_bundle, write_json)


def preflight(bundle, output, checkpoint=None, gpu_smoke=False, gpu_memory_fraction=None):
    if gpu_memory_fraction is not None and not gpu_smoke:
        raise ValueError("--gpu-memory-fraction applies to --gpu-smoke")
    print("Checking dataset files and source integrity...", flush=True)
    bundle = Path(bundle).resolve()
    manifest = verify_bundle(bundle)
    source_hash = source_integrity()
    if source_hash != manifest["source_lock_sha256"]:
        raise ValueError("Bundle was generated with a different source adaptation")
    report = dict(status="data_checks_passed", dataset_fingerprint=manifest["dataset_fingerprint"],
                  config_sha256=sha256(bundle / "baseline.yaml"), source_lock_sha256=source_hash,
                  application_sha256=application_hash(), python=platform.python_version(),
                  gpu_smoke_passed=False, training_started=False, gpu_memory_fraction=gpu_memory_fraction)
    if gpu_smoke:
        if checkpoint is None:
            raise ValueError("--gpu-smoke requires --checkpoint (a prepared initializer)")
        import torch
        report["hardware"] = require_gpu()
        report["gpu_memory_budget"] = apply_gpu_memory_limit(gpu_memory_fraction)
        if report["gpu_memory_budget"]:
            print(f"PyTorch allocator budget: {report['gpu_memory_budget']['allocator_limit_bytes']/1024**3:.2f} GiB", flush=True)
        cfg = read_cfg(bundle / "baseline.yaml")
        seed_runtime(cfg.RNG_SEED)
        import train_net  # Exercise the complete training import graph before the smoke step.
        from slowfast.models import build_model
        from slowfast.models.optimizer import construct_optimizer
        from slowfast.datasets import loader
        print("Building M2DAR and loading the verified initializer...", flush=True)
        model = build_model(cfg)
        ckpt = load_torch_checkpoint(checkpoint)
        prep = ckpt.get("preparation", {})
        if prep.get("purpose") != "generic_initializer" or prep.get("dataset_fingerprint") != manifest["dataset_fingerprint"]:
            raise ValueError("Use prepare_initializer.py for this dataset before the smoke test")
        require_shapes(model.state_dict(), ckpt["model_state"], allow_head=True)
        model.load_state_dict(ckpt["model_state"], strict=False)
        del ckpt
        val_dataset = loader.construct_loader(cfg, "val", return_dataset=True)
        # Source's return_dataset can return either dataset or (loader, dataset).
        if isinstance(val_dataset, tuple):
            val_dataset = val_dataset[-1]
        if hasattr(val_dataset, "dataset"):
            val_dataset = val_dataset.dataset
        print("Checking repeated validation decoding...", flush=True)
        first, second = val_dataset[0], val_dataset[0]
        if first[2] != second[2] or not torch.equal(first[0][0], second[0][0]):
            raise ValueError("Validation sampling is not deterministic")
        train_loader = loader.construct_loader(cfg, "train")
        print("Loading one real training batch...", flush=True)
        inputs, labels, _, _ = next(iter(train_loader))
        inputs = [x.cuda() for x in inputs]
        labels = labels.cuda()
        expected = (cfg.TRAIN.BATCH_SIZE, 3, 16, 448, 448)
        if tuple(inputs[0].shape) != expected:
            raise ValueError(f"Unexpected model input {inputs[0].shape}; expected {expected}")
        model.train()
        optimizer = construct_optimizer(model, cfg)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
        torch.cuda.reset_peak_memory_stats()
        print("Running GPU forward, loss, backward and a disposable optimizer step (fused attention)...", flush=True)
        from amp_smoke import calibrated_optimizer_step, SmokeGradientError
        def forward_loss():
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
                logits = model(inputs)
                if tuple(logits.shape) != (cfg.TRAIN.BATCH_SIZE, 16):
                    raise ValueError("Unexpected classifier output shape")
                return torch.nn.functional.cross_entropy(logits, labels)
        try:
            step = calibrated_optimizer_step(model, optimizer, scaler, forward_loss, cfg.SOLVER.CLIP_GRAD_L2NORM)
        except SmokeGradientError as error:
            report.update(status="gpu_smoke_failed_gradients", error=str(error), amp_attempts=error.attempts,
                          checkpoint_sha256=sha256(checkpoint), smoke_input_shape=list(expected))
            write_json(output, report)
            raise
        del inputs, labels
        print("Checking evaluation probabilities...", flush=True)
        model.eval()
        with torch.no_grad():
            probabilities = model([first[0][0].unsqueeze(0).cuda()])
        if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(-1), torch.ones(1, device="cuda"), atol=1e-4):
            raise ValueError("Evaluation did not return finite softmax probabilities")
        report.update(status="gpu_smoke_passed", gpu_smoke_passed=True,
                      checkpoint_sha256=sha256(checkpoint), smoke_input_shape=list(expected),
                      smoke_loss=step["loss"], gradient_norm=step["gradient_norm"],
                      amp_scale_used=step["scale_used"], amp_attempts=step["attempts"],
                      skipped_amp_overflow_steps=step["skipped_overflow_steps"],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      validation_repeat_identical=True,
                      attention_backend_policy="fused CUDA SDPA, no math fallback; same pooled-attention equation",
                      smoke_note="One finite disposable optimizer step after bounded AMP calibration; no weights saved, no experiment epoch run")
        report["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    write_json(output, report)
    print(report["status"], "->", output)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--gpu-smoke", action="store_true")
    p.add_argument("--gpu-memory-fraction", type=float, help="Optional Torch allocator fraction; must also be used for training")
    a = p.parse_args()
    preflight(a.bundle, a.out, a.checkpoint, a.gpu_smoke, a.gpu_memory_fraction)
