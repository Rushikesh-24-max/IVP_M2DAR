"""Evaluate every held-out clip and report classification metrics with saved predictions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import json

from common import (VIEWS, load_torch_checkpoint, read_cfg, read_json, read_jsonl, require_gpu,
                    require_shapes, seed_runtime, sha256, verify_bundle, write_json)
from metrics import classification_metrics


def evaluate(bundle, checkpoint, out):
    import torch
    bundle, checkpoint, out = Path(bundle).resolve(), Path(checkpoint).resolve(), Path(out).resolve()
    manifest = verify_bundle(bundle)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new evaluation output directory")
    require_gpu()
    cfg = read_cfg(bundle / "baseline.yaml")
    seed_runtime(cfg.RNG_SEED)
    from slowfast.models import build_model
    from slowfast.datasets import loader
    state = load_torch_checkpoint(checkpoint)
    if checkpoint.name == "best.pyth":
        run = checkpoint.parent
    else:
        run = checkpoint.parent.parent
    run_meta = read_json(run / "run.json")
    if run_meta["dataset_fingerprint"] != manifest["dataset_fingerprint"]:
        raise ValueError("Checkpoint training split does not match this evaluation bundle")
    if "preparation" in state:
        raise ValueError("Evaluate a trained checkpoint, not the generic initializer")
    model = build_model(cfg)
    require_shapes(model.state_dict(), state["model_state"])
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    index = read_jsonl(bundle / "val_index.jsonl")
    predictions = {}
    with torch.no_grad():
        for inputs, labels, ids, _ in loader.construct_loader(cfg, "val"):
            scores = model([x.cuda(non_blocking=True) for x in inputs]).cpu().tolist()
            for idx, label, values in zip(ids.tolist(), labels.tolist(), scores):
                if idx in predictions or label != index[idx]["label"]:
                    raise ValueError("Duplicate sample or label mismatch in evaluation")
                predictions[idx] = dict(index[idx], probabilities=values)
    if set(predictions) != set(range(len(index))):
        raise ValueError("Some validation clips were not evaluated")
    rows = [predictions[i] for i in range(len(index))]
    def score(values):
        return classification_metrics([r["label"] for r in values], [r["probabilities"] for r in values])
    result = dict(single_view_clips=score(rows), by_camera={v: score([r for r in rows if r["view"] == v]) for v in VIEWS})
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    fused = []
    for gid, views in groups.items():
        if len(views) != 3 or len({r["label"] for r in views}) != 1:
            raise ValueError(f"Invalid validation triplet: {gid}")
        fused.append(dict(group_id=gid, label=views[0]["label"],
                          probabilities=[sum(r["probabilities"][i] for r in views) / 3 for i in range(16)]))
    result.update(equal_weight_three_camera_classification=score(fused),
                  dataset_fingerprint=manifest["dataset_fingerprint"], checkpoint_sha256=sha256(checkpoint),
                  protocol="one deterministic center window/crop per clip; camera averaging is a classification diagnostic, not temporal Election or official challenge scoring")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", result)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    confusion = result["single_view_clips"]["confusion_matrix"]
    (out / "confusion_matrix.csv").write_text("true_class," + ",".join(f"pred_{i}" for i in range(16)) + "\n" +
        "".join(str(i) + "," + ",".join(map(str, row)) + "\n" for i, row in enumerate(confusion)), encoding="utf-8")
    print("Evaluated", len(rows), "clips from", len(fused), "triplets ->", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluate(a.bundle, a.checkpoint, a.out)
