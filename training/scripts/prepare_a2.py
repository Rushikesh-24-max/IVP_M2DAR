"""Create portable A2 inference lists without generating labels or copying media."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
import csv

from common import VIEWS, read_json, read_jsonl, sha256, write_json


def original_relative(value):
    path = PureWindowsPath(value)
    parts = path.parts[1:] if path.drive else path.parts
    if not parts or parts[0] != "A2" or ".." in parts or any(":" in p for p in parts):
        raise ValueError(f"Unexpected original A2 path: {value}")
    return PurePosixPath(*parts).as_posix()


def prepare(data_root, mode, out):
    root, out = Path(data_root).resolve(), Path(out).resolve()
    package = root / "A2_preprocessed"
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty A2 bundle directory")
    if mode not in ("aligned", "original"):
        raise ValueError(mode)
    records = read_jsonl(package / "inference_groups.jsonl") if mode == "aligned" else read_json(package / "original_recordings.json")
    mapping, files, ids, basenames = [], [], set(), set()
    for record in records:
        identifier = record["group_id"] if mode == "aligned" else record["video_id"]
        if str(identifier) in ids:
            raise ValueError("Duplicate A2 identifier")
        ids.add(str(identifier))
        names = []
        if mode == "aligned" and (record["label"] is not None or record["split"] != "inference"):
            raise ValueError("A2 must remain unlabelled inference data")
        for view in VIEWS:
            info = record["views"][view]
            if mode == "aligned":
                relative = PurePosixPath(info["relative_output_path"])
                if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "clips":
                    raise ValueError("Invalid A2 clip path")
                relative = (PurePosixPath("A2_preprocessed") / relative).as_posix()
                size = record["validation"]["output_file_stats"][view]["size_bytes"]
            else:
                relative = original_relative(info["path"])
                size = info["source_size_bytes"]
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Missing/size-changed A2 media: {path}")
            if path.name in basenames:
                raise ValueError(f"A2 basename collision would overwrite scores: {path.name}")
            basenames.add(path.name)
            names.append(path.name)
            files.append(relative)
        mapping.append([identifier] + names)
    expected = 415 if mode == "aligned" else 30
    if len(records) != expected:
        raise ValueError(f"Expected {expected} A2 records; found {len(records)}")
    supplied = package / "m2dar" / ("A2_segments_video_ids.csv" if mode == "aligned" else "A2_original_video_ids.csv")
    with supplied.open(newline="", encoding="utf-8-sig") as f:
        official_rows = list(csv.reader(f))[1:]
    if sorted(official_rows) != sorted([[str(x) for x in r] for r in mapping]):
        raise ValueError("Prepared A2 mapping differs from the existing ID map")
    out.mkdir(parents=True, exist_ok=True)
    (out / "videos.lst").write_text("\n".join(files) + "\n", encoding="utf-8")
    with (out / "video_ids.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", *VIEWS])
        writer.writerows(mapping)
    coverage = read_json(package / "inference_coverage.json")
    total = sum(r["original_dashboard_duration_s"] for r in coverage)
    included = sum(sum(b-a for a, b in r["included_reference_intervals_s"] ) for r in coverage)
    report = dict(mode=mode, data_root=str(root), recordings_or_segments=len(records), video_files=len(files),
                  supervised_labels=False, aligned_reference_coverage=included/total,
                  source_metadata_sha256=sha256(package / ("inference_groups.jsonl" if mode == "aligned" else "original_recordings.json")),
                  artifacts={name: sha256(out / name) for name in ("videos.lst", "video_ids.csv")},
                  limitation=("Partial aligned coverage only; restore timestamps and retain unsupported intervals. Not a full A2 evaluation."
                              if mode == "aligned" else "Full original streams; camera offsets are not corrected by this baseline route."))
    write_json(out / "a2.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--mode", choices=["aligned", "original"], default="aligned")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(prepare(a.data_root, a.mode, a.out))
