"""Start or resume the released M2DAR trainer using the verified local data bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

from common import (application_hash, apply_gpu_memory_limit, load_torch_checkpoint, read_cfg, read_json, require_gpu, require_shapes,
                    seed_runtime, sha256, source_integrity, verify_bundle, write_json)


def validate_smoke(report, manifest, bundle, checkpoint, gpu_memory_fraction=None):
    expected = dict(dataset_fingerprint=manifest["dataset_fingerprint"],
                    config_sha256=sha256(Path(bundle) / "baseline.yaml"),
                    source_lock_sha256=source_integrity(), application_sha256=application_hash(),
                    checkpoint_sha256=sha256(checkpoint), gpu_smoke_passed=True,
                    gpu_memory_fraction=gpu_memory_fraction)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"GPU smoke test does not match current inputs: {key}; rerun preflight")


def launch(bundle, run, initializer=None, preflight_report=None, resume=None, dry_run=False, keep_last_checkpoints=2, gpu_memory_fraction=None):
    if type(keep_last_checkpoints) is not int or keep_last_checkpoints < 1:
        raise ValueError("Retain at least one periodic restart checkpoint")
    bundle, run = Path(bundle).resolve(), Path(run).resolve()
    manifest = verify_bundle(bundle)
    if manifest["source_lock_sha256"] != source_integrity():
        raise ValueError("Source changed since data preparation")
    if resume:
        checkpoint = Path(resume).resolve()
        record = read_json(run / "run.json")
        if record.get("gpu_memory_fraction") != gpu_memory_fraction:
            raise ValueError("Resume must use the recorded --gpu-memory-fraction")
        for key, value in (("dataset_fingerprint", manifest["dataset_fingerprint"]),
                           ("config_sha256", sha256(bundle / "baseline.yaml")),
                           ("application_sha256", application_hash()),
                           ("source_lock_sha256", source_integrity())):
            if record[key] != value:
                raise ValueError(f"Cannot resume a different experiment: {key}")
        if checkpoint.parent != run / "checkpoints" or not checkpoint.is_file():
            raise ValueError("Resume needs a periodic checkpoint in this run's checkpoints directory, not best.pyth")
        if sha256(run / "resolved.yaml") != record["resolved_config_sha256"]:
            raise ValueError("Original resolved configuration was changed")
    else:
        if not initializer or not preflight_report:
            raise ValueError("Fresh training requires --initializer and a passed --preflight-report")
        checkpoint = Path(initializer).resolve()
        smoke = read_json(preflight_report)
        validate_smoke(smoke, manifest, bundle, checkpoint, gpu_memory_fraction)
        if run.exists() and any(run.iterdir()):
            raise ValueError("Fresh training needs a new/empty run directory; use --resume for an existing run")
    if dry_run:
        print(json.dumps(dict(action="resume" if resume else "fresh_training", bundle=str(bundle),
                              run=str(run), checkpoint=str(checkpoint), epochs=200,
                              keep_last_checkpoints=record.get("keep_last_checkpoints") if resume else keep_last_checkpoints,
                              gpu_memory_fraction=gpu_memory_fraction,
                              executed=False), indent=2))
        return
    hardware = require_gpu()
    memory_budget = apply_gpu_memory_limit(gpu_memory_fraction)
    if not resume and hardware != smoke["hardware"]:
        raise ValueError("GPU/runtime changed since smoke test; rerun preflight on this server")
    cfg = read_cfg(bundle / "baseline.yaml")
    cfg.OUTPUT_DIR = str(run)
    cfg.TENSORBOARD.LOG_DIR = str(run / "tb_log")
    cfg.TRAIN.CHECKPOINT_FILE_PATH = str(checkpoint)
    cfg.TRAIN.AUTO_RESUME = False
    cfg.TRAIN.CHECKPOINT_EPOCH_RESET = not bool(resume)
    state = load_torch_checkpoint(checkpoint)
    if resume:
        if "optimizer_state" not in state or type(state.get("epoch")) is not int:
            raise ValueError("Checkpoint lacks optimizer/epoch state; cannot resume")
        if state["epoch"] + 1 >= cfg.SOLVER.MAX_EPOCH:
            raise ValueError("This run already completed the configured epochs")
        import yaml
        saved_cfg, current_cfg = yaml.safe_load(state["cfg"]), yaml.safe_load(cfg.dump())
        for key in ("MODEL", "MVIT", "DATA", "SOLVER", "BN", "AUG", "MIXUP", "RNG_SEED",
                    "NUM_GPUS", "NUM_SHARDS", "DATA_LOADER", "TEST"):
            if saved_cfg[key] != current_cfg[key]:
                raise ValueError(f"Resume checkpoint configuration differs: {key}")
        for key in current_cfg["TRAIN"]:
            if key not in ("CHECKPOINT_FILE_PATH", "AUTO_RESUME", "CHECKPOINT_EPOCH_RESET") and saved_cfg["TRAIN"][key] != current_cfg["TRAIN"][key]:
                raise ValueError(f"Resume checkpoint training configuration differs: {key}")
        from slowfast.models import build_model
        cpu_cfg = cfg.clone()
        cpu_cfg.NUM_GPUS = 0
        inspection_model = build_model(cpu_cfg)
        require_shapes(inspection_model.state_dict(), state["model_state"])
        del inspection_model
    elif state.get("preparation", {}).get("purpose") != "generic_initializer":
        raise ValueError("Fresh training requires an audited generic initializer")
    del state
    run.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    if not resume:
        (run / "resolved.yaml").write_text(cfg.dump(), encoding="utf-8")
        record = dict(run_id=str(uuid.uuid4()), created_utc=timestamp, dataset_fingerprint=manifest["dataset_fingerprint"],
                      config_sha256=sha256(bundle / "baseline.yaml"), resolved_config_sha256=sha256(run / "resolved.yaml"),
                      application_sha256=application_hash(), source_lock_sha256=source_integrity(),
                      initializer_sha256=sha256(checkpoint), validation_clips=manifest["summary"]["val"]["clips"],
                      hardware=hardware, bundle=str(bundle), status="running", preflight=smoke,
                      keep_last_checkpoints=keep_last_checkpoints, gpu_memory_fraction=gpu_memory_fraction,
                      gpu_memory_budget=memory_budget)
        write_json(run / "run.json", record)
    else:
        write_json(run / "last_resume.json", dict(utc=timestamp, checkpoint=str(checkpoint),
                                                  sha256=sha256(checkpoint), hardware=hardware))
        (run / "resume.yaml").write_text(cfg.dump(), encoding="utf-8")
    seed_runtime(cfg.RNG_SEED)
    from train_net import train
    try:
        train(cfg)
    except BaseException:
        record["status"] = "interrupted_or_failed"
        write_json(run / "run.json", record)
        raise
    record.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(run / "run.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--run", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initializer")
    mode.add_argument("--resume", help="Periodic checkpoint, with optimizer state, from this exact run")
    p.add_argument("--preflight-report")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-last-checkpoints", type=int, default=2,
                   help="Fresh-run periodic checkpoint retention; resume reuses the recorded policy")
    p.add_argument("--gpu-memory-fraction", type=float, help="Match the smoke test's optional Torch allocator fraction")
    a = p.parse_args()
    launch(a.bundle, a.run, a.initializer, a.preflight_report, a.resume, a.dry_run, a.keep_last_checkpoints, a.gpu_memory_fraction)
