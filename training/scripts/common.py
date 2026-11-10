"""Shared paths and integrity checks. This module has no ML dependencies."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "M2DAR"
REVISION = "8fc6a3f76fc47207c4fa072dff7259f0debb5472"
VIEWS = ("dashboard", "rearview", "right_side_window")
PARTS = ("A1_preprocessed", "A1_2_preprocessed", "A1_3_preprocessed", "A1_4_preprocessed")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_integrity():
    lock = read_json(ROOT / "source_lock.json")
    expected = dict(lock["upstream_files"])
    expected.update(lock.get("adapted_files", {}))
    optional_figures = {".github/framework.png", *(f".github/vis{i}.png" for i in range(1, 5))}
    bad = []
    for name, digest in expected.items():
        path = VENDOR / name
        if name in optional_figures and not path.exists():
            continue
        if not path.is_file() or sha256(path) != digest:
            bad.append(name)
    if bad:
        raise ValueError("Source files differ from source_lock.json: " + ", ".join(bad))
    return sha256(ROOT / "source_lock.json")


def a1_relative_path(relative, package):
    """Map combined-manifest paths to the portable parent directory layout."""
    value = PurePosixPath(relative.replace("\\", "/"))
    if package not in PARTS or value.parts[:2] != ("..", package):
        raise ValueError(f"Unexpected A1 path/package: {relative!r}, {package!r}")
    result = PurePosixPath(*value.parts[1:])
    if ".." in result.parts or result.is_absolute() or any(c in str(result) for c in "\t\r\n:"):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    return result.as_posix()


def verify_bundle(bundle, check_files=True):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / "dataset.json")
    for name, digest in manifest["artifacts"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Prepared bundle modified: {name}; rebuild in a new directory")
    if check_files:
        root = Path(manifest["data_root"])
        for split in ("train", "val"):
            for row in read_jsonl(bundle / f"{split}_index.jsonl"):
                path = root / row["path"]
                if not path.is_file() or path.stat().st_size != row["size_bytes"]:
                    raise ValueError(f"Missing/size-changed clip: {path}")
    return manifest


def activate_source():
    source_integrity()
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(VENDOR / "tools"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def read_cfg(path):
    activate_source()
    from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(path).resolve()))
    return assert_and_infer_cfg(cfg)


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA GPU is required. No training has started.")
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory}


def apply_gpu_memory_limit(fraction):
    """Limit the Torch allocator; CUDA context/library allocations are additional."""
    if fraction is None:
        return None
    import math
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0, 1]")
    import torch
    free, total = torch.cuda.mem_get_info(0)
    limit = int(total * fraction)
    if free < limit + 1024**3:
        raise RuntimeError("Not enough currently free GPU memory for the requested allocator budget plus 1 GiB headroom")
    torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)
    return dict(fraction=float(fraction), allocator_limit_bytes=limit, free_bytes_at_start=free,
                note="Limits PyTorch's allocator; CUDA context/library allocations are additional. Not an exclusive GPU reservation.")


def application_hash():
    return stable_hash({p.name: sha256(p) for p in sorted((ROOT / "scripts").glob("*.py"))})


def seed_runtime(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_shapes(model_state, checkpoint_state, allow_head=False):
    ignored = ("head.",) if allow_head else ()
    bad = [name for name, value in model_state.items()
           if not name.startswith(ignored) and
           (name not in checkpoint_state or tuple(value.shape) != tuple(checkpoint_state[name].shape))]
    extra = [name for name in checkpoint_state if name not in model_state and not name.startswith(ignored)]
    if bad or extra:
        raise ValueError(f"Incompatible checkpoint. Missing/mismatched: {bad}; unexpected: {extra}")


def load_torch_checkpoint(path):
    import torch
    # Package checkpoints contain tensors and primitive metadata only.
    result = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or not isinstance(result.get("model_state"), dict):
        raise ValueError("Expected a PySlowFast checkpoint containing model_state")
    return result
