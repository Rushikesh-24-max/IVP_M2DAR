"""Run released sliding-window inference and Election on a prepared A2 list."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys

from common import (ROOT, VENDOR, load_torch_checkpoint, read_cfg, read_json, require_gpu,
                    require_shapes, sha256, source_integrity, verify_bundle, write_json)


def infer(bundle, a2_bundle, checkpoint, out, workers=0, dry_run=False):
    bundle, a2_bundle, checkpoint, out = map(lambda p: Path(p).resolve(), (bundle, a2_bundle, checkpoint, out))
    manifest = verify_bundle(bundle)
    a2 = read_json(a2_bundle / "a2.json")
    source_integrity()
    for name, digest in a2["artifacts"].items():
        if sha256(a2_bundle / name) != digest:
            raise ValueError("A2 inference list was modified")
    if a2["data_root"] != manifest["data_root"]:
        raise ValueError("A1 and A2 bundles must refer to the same relocated dataset root")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new inference directory to avoid mixing checkpoints/modes")
    paths = [Path(a2["data_root"]) / p for p in (a2_bundle / "videos.lst").read_text().splitlines()]
    if any(not p.is_file() for p in paths):
        raise ValueError("Missing A2 media after list preparation")
    run = checkpoint.parent if checkpoint.name == "best.pyth" else checkpoint.parent.parent
    if read_json(run / "run.json")["dataset_fingerprint"] != manifest["dataset_fingerprint"]:
        raise ValueError("Inference checkpoint does not belong to this research split")
    commands = [
        [sys.executable, str(VENDOR / "tools/inference/driver_action_rec.py"),
         "--video_lst", str(a2_bundle / "videos.lst"), "--video_dir", a2["data_root"],
         "--model_path", str(checkpoint), "--out_dir", str(out / "scores"),
         "--pyslowfast_cfg", str(bundle / "baseline.yaml"), "--model_dataset", "aicity", "--model_type", "mvit",
         "--frame_length", "16", "--frame_stride", "4", "--proposal_length", "64", "--proposal_stride", "16",
         "--frame_size", "448", "--batch_size", "1", "--num_cpu_workers", str(workers), "--gpu_id", "0"],
        [sys.executable, str(VENDOR / "tools/inference/election.py"),
         "--pred_pickle_path", str(out / "scores"), "--vid_csv", str(a2_bundle / "video_ids.csv"),
         "--threshold_path", str(VENDOR / "configs"), "--output_path", str(out / "election")],
    ]
    if a2["mode"] == "aligned":
        commands.append([sys.executable, str(ROOT / "scripts/restore_a2_times.py"),
                         "--input", str(out / "election/submit.txt"), "--output", str(out / "restored_candidates.jsonl"),
                         "--root", str(Path(a2["data_root"]) / "A2_preprocessed")])
    if dry_run:
        import json
        print(json.dumps(commands, indent=2))
        return
    require_gpu()
    cfg = read_cfg(bundle / "baseline.yaml")
    cfg.NUM_GPUS = 0
    from slowfast.models import build_model
    model = build_model(cfg)
    state = load_torch_checkpoint(checkpoint)
    require_shapes(model.state_dict(), state["model_state"])
    del model, state
    out.mkdir(parents=True, exist_ok=True)
    record = dict(checkpoint_sha256=sha256(checkpoint), dataset_fingerprint=manifest["dataset_fingerprint"],
                  source_lock_sha256=source_integrity(), a2=a2, commands=commands, status="running",
                  thresholds="Unmodified released Election thresholds; not tuned or validated for 2024",
                  metrics="No local A2 ground truth: accuracy/localization score unavailable")
    write_json(out / "inference.json", record)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(VENDOR), str(ROOT / "scripts"), env.get("PYTHONPATH", "")])
    env.setdefault("MPLBACKEND", "Agg")
    try:
        for command in commands:
            subprocess.run(command, cwd=VENDOR, env=env, check=True)
    except BaseException:
        record["status"] = "interrupted_or_failed"
        write_json(out / "inference.json", record)
        raise
    record["status"] = "completed"
    write_json(out / "inference.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--a2-bundle", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=0, help="Default avoids sharing an open Decord reader across processes")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    infer(a.bundle, a.a2_bundle, a.checkpoint, a.out, a.workers, a.dry_run)
