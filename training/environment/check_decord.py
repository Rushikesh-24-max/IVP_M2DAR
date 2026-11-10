"""Test the real decoder, then repair only Decord 0.6.0's known wheel-tag defect."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD_TAG = "Tag: cp36-cp36m-manylinux2010_x86_64"
NEW_TAG = "Tag: py3-none-manylinux2010_x86_64"


def repaired_contents(wheel_bytes, record_bytes, record_name):
    text = wheel_bytes.decode("utf-8")
    tags = [line for line in text.splitlines() if line.startswith("Tag:")]
    if tags == [NEW_TAG]:
        return wheel_bytes, record_bytes, False
    if tags != [OLD_TAG]:
        raise ValueError(f"Unrecognized wheel tags; no repair applied: {tags}")
    new_wheel = text.replace(OLD_TAG, NEW_TAG).encode("utf-8")
    rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"))))
    matching = [row for row in rows if row[0] == record_name]
    if len(matching) != 1:
        raise ValueError("Cannot identify the WHEEL entry in RECORD; no repair applied")
    encoded = base64.urlsafe_b64encode(hashlib.sha256(new_wheel).digest()).rstrip(b"=").decode("ascii")
    matching[0][1:] = ["sha256=" + encoded, str(len(new_wheel))]
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return new_wheel, buffer.getvalue().encode("utf-8"), True


def main(video):
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the project virtual environment first")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This repair is only for the reported Linux x86_64 environment")
    import decord
    import numpy as np
    from pip._vendor.packaging.tags import Tag, sys_tags
    if Tag("py3", "none", "manylinux2010_x86_64") not in set(sys_tags()):
        raise RuntimeError("The published wheel filename is not compatible with this interpreter")
    dist = metadata.distribution("decord")
    if dist.version != "0.6.0":
        raise RuntimeError("Only the published Decord 0.6.0 wheel is covered by this repair")
    video = Path(video).resolve()
    reader = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
    if len(reader) < 64 or abs(reader.get_avg_fps() - 30) > .01:
        raise RuntimeError("Unexpected smoke-clip frame count or frame rate")
    frames = reader.get_batch(np.linspace(0, 63, 16).astype("int64")).asnumpy()
    if frames.shape != (16, 540, 960, 3) or frames.dtype != np.uint8:
        raise RuntimeError(f"Unexpected decoded frames: {frames.shape}, {frames.dtype}")
    print("Real H.264 decode passed:", frames.shape, frames.dtype, flush=True)
    entries = [entry for entry in dist.files if str(entry).endswith(".dist-info/WHEEL")]
    if len(entries) != 1:
        raise RuntimeError("Could not identify the installed WHEEL metadata")
    entry = entries[0]
    wheel_path = Path(dist.locate_file(entry)).resolve()
    record_path = wheel_path.with_name("RECORD").resolve()
    prefix = Path(sys.prefix).resolve()
    if not wheel_path.is_relative_to(prefix) or not record_path.is_relative_to(prefix):
        raise RuntimeError("Metadata is outside the active environment; no repair applied")
    old_wheel, old_record = wheel_path.read_bytes(), record_path.read_bytes()
    new_wheel, new_record, changed = repaired_contents(old_wheel, old_record, str(entry).replace("\\", "/"))
    checks = ROOT / "checks"
    checks.mkdir(exist_ok=True)
    if changed:
        # Keep originals outside the installation; binaries and Python package files are untouched.
        backup = checks / "decord_metadata_backup.json"
        if not backup.exists():
            backup.write_text(json.dumps(dict(wheel_path=str(wheel_path), record_path=str(record_path),
                wheel_base64=base64.b64encode(old_wheel).decode(), record_base64=base64.b64encode(old_record).decode()), indent=2), encoding="utf-8")
        try:
            wheel_path.write_bytes(new_wheel)
            record_path.write_bytes(new_record)
        except BaseException:
            wheel_path.write_bytes(old_wheel)
            record_path.write_bytes(old_record)
            raise
    report = dict(decord_version=dist.version, python=sys.version, real_h264_decode_passed=True,
                  frames_shape=list(frames.shape), video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                  metadata_tag_corrected=changed, native_binaries_modified=False,
                  upstream_issue="https://github.com/dmlc/decord/issues/356")
    (checks / "decord_runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wheel tag corrected and RECORD hash updated" if changed else "Wheel tag already correct")
    print("Run: python -m pip check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "checks/decord_sample.mp4"))
    main(parser.parse_args().video)
