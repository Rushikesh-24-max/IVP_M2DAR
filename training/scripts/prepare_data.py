"""Validate the existing A1 split and create a portable, media-free run bundle."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from common import (PARTS, VIEWS, VENDOR, a1_relative_path, read_json, read_jsonl,
                    sha256, source_integrity, stable_hash, write_json)


def collect(data_root):
    root = Path(data_root).resolve()
    combined = root / "M2DAR_preprocessed"
    ready = read_json(combined / "READY.json")
    rows = {s: read_jsonl(combined / f"{s}_groups.jsonl") for s in ("train", "val")}
    indexes, summary = {}, {}
    ids, seen_paths = set(), set()
    excluded = set(ready["A2_drivers_excluded"])
    for split, groups in rows.items():
        index = []
        drivers, counts = set(), Counter()
        for group in groups:
            gid = group["group_id"]
            if gid in ids or group["split"] != split:
                raise ValueError(f"Duplicate group or wrong split: {gid}")
            ids.add(gid)
            label, driver = group["label"], str(group["driver"])
            if type(label) is not int or not 0 <= label < 16 or driver in excluded:
                raise ValueError(f"Invalid label/A2 driver in supervised split: {gid}")
            if set(group["views"]) != set(VIEWS) or group["fps"] != 30 or group["frame_count"] < 16:
                raise ValueError(f"Invalid views/frame specification: {gid}")
            if group["validation"]["status"] != "ready":
                raise ValueError(f"Group not cleared by preprocessing: {gid}")
            drivers.add(driver)
            counts[label] += 1
            for view in VIEWS:
                relative = a1_relative_path(group["views"][view]["relative_output_path"], group["source_package"])
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or relative in seen_paths:
                    raise ValueError(f"Duplicate/outside-root clip: {path}")
                seen_paths.add(relative)
                expected = group["validation"]["output_file_stats"][view]["size_bytes"]
                if not path.is_file() or path.stat().st_size != expected:
                    raise ValueError(f"Missing or changed video: {path}")
                health = group["validation"]["video_checks"][view]
                if health["frames"] != group["frame_count"] or health["corrupt_frames"]:
                    raise ValueError(f"Invalid video audit: {path}")
                index.append(dict(group_id=gid, driver=driver, label=label, view=view,
                                  path=relative, size_bytes=expected, frames=group["frame_count"]))
        if drivers != set(ready[f"{split}_drivers"]) or set(counts) != set(range(16)):
            raise ValueError(f"Driver split/class coverage changed in {split}")
        if len(groups) != ready[f"{split}_triplets"] or len(index) != ready[f"{split}_csv_entries"]:
            raise ValueError(f"Count mismatch in {split}")
        original = []
        for line in (combined / "m2dar" / f"{split}.csv").read_text().splitlines():
            path, label = line.rsplit(" ", 1)
            original.append((path.replace("\\", "/").removeprefix("../"), int(label)))
        if Counter(original) != Counter((r["path"], r["label"]) for r in index):
            raise ValueError(f"CSV and group annotations disagree in {split}")
        indexes[split] = index
        summary[split] = dict(groups=len(groups), clips=len(index), drivers=sorted(drivers),
                              class_groups=dict(sorted(counts.items())),
                              short_groups=sum(g["frame_count"] < 64 for g in groups))
    if set(summary["train"]["drivers"]) & set(summary["val"]["drivers"]):
        raise ValueError("Driver leakage between train and validation")
    if {g["source_package"] for values in rows.values() for g in values} != set(PARTS):
        raise ValueError("Expected all four prepared A1 parts")
    all_manifest = read_jsonl(combined / "training_manifest.jsonl")
    canonical = {g["group_id"]: g for g in all_manifest}
    if len(canonical) != len(all_manifest) or canonical != {g["group_id"]: g for v in rows.values() for g in v}:
        raise ValueError("Combined manifest differs from split manifests")
    return indexes, summary


def prepare(data_root, out, batch_size=3, workers=4, amp=False):
    import yaml
    source_hash = source_integrity()
    data_root, out = Path(data_root).resolve(), Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new/empty bundle directory; existing experiment metadata is preserved")
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid batch size/worker count")
    indexes, summary = collect(data_root)
    out.mkdir(parents=True, exist_ok=True)
    for split, index in indexes.items():
        (out / f"{split}.csv").write_text("".join(f"{r['path']}\t{r['label']}\n" for r in index), encoding="utf-8")
        (out / f"{split}_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index), encoding="utf-8")
    cfg = yaml.safe_load((VENDOR / "configs/MVITV2_S_16x4_448.yaml").read_text())
    cfg["DATA"].update(PATH_PREFIX=data_root.as_posix(), PATH_TO_DATA_DIR=out.as_posix(),
                       PATH_LABEL_SEPARATOR="\t")
    cfg["TRAIN"].update(BATCH_SIZE=batch_size, CHECKPOINT_FILE_PATH="", AUTO_RESUME=False,
                        CHECKPOINT_EPOCH_RESET=True, CHECKPOINT_PERIOD=10,
                        EVAL_PERIOD=10, MIXED_PRECISION=amp)
    cfg["TEST"].update(ENABLE=False, BATCH_SIZE=batch_size, NUM_ENSEMBLE_VIEWS=1)
    cfg["DATA_LOADER"].update(NUM_WORKERS=workers)
    cfg.update(OUTPUT_DIR=(out / "UNUSED_set_by_launcher").as_posix(), RNG_SEED=42, LOG_PERIOD=20)
    cfg["TENSORBOARD"]["LOG_DIR"] = "tb_log"
    (out / "baseline.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    artifacts = {name: sha256(out / name) for name in
                 ("train.csv", "val.csv", "train_index.jsonl", "val_index.jsonl", "baseline.yaml")}
    manifest = dict(schema_version=1, data_root=str(data_root), summary=summary,
                    dataset_fingerprint=stable_hash(indexes), source_lock_sha256=source_hash,
                    input_metadata_sha256={name: sha256(data_root / "M2DAR_preprocessed" / name) for name in
                                           ("READY.json", "training_manifest.jsonl", "train_groups.jsonl", "val_groups.jsonl")},
                    artifacts=artifacts, media_verification="existence, byte sizes, prior complete decode audits; not full media hashes",
                    training_started=False, protocol="single-view M2DAR S baseline; deterministic center validation; 32/8 drivers")
    write_json(out / "dataset.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, help="Parent of M2DAR_preprocessed and the four A1 prepared folders")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="Explicit deviation from the FP32 released configuration")
    a = p.parse_args()
    result = prepare(a.data_root, a.out, a.batch_size, a.workers, a.amp)
    print(json.dumps(result["summary"], indent=2))
    print("Prepared bundle. Training has not started.")
