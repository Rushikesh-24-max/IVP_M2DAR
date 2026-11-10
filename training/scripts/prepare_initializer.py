"""Audit and normalize a generic pretrained checkpoint; never starts training."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import (load_torch_checkpoint, read_cfg, read_json, require_shapes,
                    sha256, verify_bundle, write_json)


def audit_provenance(path, provenance):
    if provenance.get("kind") != "generic_action_pretraining":
        raise ValueError("Initializer must be generic action pretraining, not an AICity/SynDD2 fine-tuned model")
    if provenance.get("aicity_or_syndd2_exposure") is not False:
        raise ValueError("Document that initializer training excluded AICity/SynDD2 drivers")
    for field in ("source_url", "training_datasets", "evidence"):
        if not provenance.get(field) or "REPLACE" in str(provenance[field]):
            raise ValueError(f"Complete the initializer provenance field: {field}")
    digest = sha256(path)
    if provenance.get("sha256", "").lower() != digest:
        raise ValueError("Initializer SHA256 differs from the declared provenance")
    return digest


def prepare(bundle, checkpoint, provenance_path, output, resize_spatial=False):
    import torch
    import torch.nn.functional as F
    manifest = verify_bundle(bundle)
    provenance = read_json(provenance_path)
    digest = audit_provenance(checkpoint, provenance)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = read_cfg(Path(bundle) / "baseline.yaml")
    cfg.NUM_GPUS = 0
    from slowfast.models import build_model
    model = build_model(cfg)
    expected = model.state_dict()
    raw = load_torch_checkpoint(checkpoint)["model_state"]
    state = {}
    for key, value in raw.items():
        name = key.removeprefix("module.")
        if name in state:
            raise ValueError(f"Duplicate normalized checkpoint key {name}")
        if not name.startswith("head."):
            state[name] = value
    operations = ["remove optional module. prefix", "discard generic classification head; initialize a new 16-class head"]
    key = "pos_embed_spatial"
    if key in state and key in expected and state[key].shape != expected[key].shape:
        if not resize_spatial:
            raise ValueError("Spatial position embeddings differ. Review the dimensions, then explicitly use --resize-spatial-position.")
        old, target = state[key], expected[key]
        old_side, new_side = math.isqrt(old.shape[1]), math.isqrt(target.shape[1])
        if old.ndim != 3 or old.shape[0] != 1 or old_side ** 2 != old.shape[1] or new_side ** 2 != target.shape[1] or old.shape[2] != target.shape[2]:
            raise ValueError("Cannot resize non-square or incompatible spatial position embeddings")
        value = old.float().reshape(1, old_side, old_side, old.shape[2]).permute(0, 3, 1, 2)
        value = F.interpolate(value, size=(new_side, new_side), mode="bicubic", align_corners=False)
        state[key] = value.permute(0, 2, 3, 1).reshape(target.shape).to(old.dtype)
        operations.append(f"bicubic spatial position interpolation {old_side}x{old_side} -> {new_side}x{new_side}")
    require_shapes(expected, state, allow_head=True)
    if not all(torch.isfinite(t).all().item() for t in state.values()):
        raise ValueError("Non-finite initializer tensors")
    metadata = dict(purpose="generic_initializer", source_sha256=digest, provenance=provenance,
                    operations=operations, backbone_coverage=1.0,
                    dataset_fingerprint=manifest["dataset_fingerprint"],
                    source_lock_sha256=manifest["source_lock_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state=state, preparation=metadata), output)
    metadata["prepared_sha256"] = sha256(output)
    write_json(output.with_suffix(output.suffix + ".json"), metadata)
    print("Initializer prepared:", output)
    print("All backbone tensors match. No training was performed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resize-spatial-position", action="store_true")
    a = p.parse_args()
    prepare(a.bundle, a.checkpoint, a.provenance, a.output, a.resize_spatial_position)
