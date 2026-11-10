"""Apply the small, recorded adaptation to the exact upstream source snapshot."""
import difflib
from common import ROOT, VENDOR, read_json, sha256, write_json


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Patch context not unique: {before[:90]!r}")
    return text.replace(before, after, 1)


def main():
    lock = read_json(ROOT / "source_lock.json")
    if lock.get("adapted_files"):
        from common import source_integrity
        source_integrity()
        print("Recorded patches already applied")
        return
    changes = {}
    path = "slowfast/datasets/aicity.py"
    text = (VENDOR / path).read_text()
    text = replace_once(text, 'if self.mode in ["train", "val"]:\n            # -1 indicates random sampling.',
                        'if self.mode in ["train"]:\n            # -1 indicates random sampling.')
    text = replace_once(text, '        elif self.mode in ["test"]:\n            temporal_sample_index = (',
                        '        elif self.mode == "val":\n'
                        '            # Prepared-data protocol: one center time window and center crop.\n'
                        '            temporal_sample_index = 0\n'
                        '            spatial_sample_index = 1\n'
                        '            min_scale = max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]\n'
                        '            crop_size = self.cfg.DATA.TEST_CROP_SIZE\n'
                        '        elif self.mode in ["test"]:\n            temporal_sample_index = (')
    text = replace_once(text, '                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,\n                video_meta=',
                        '                1 if self.mode == "val" else self.cfg.TEST.NUM_ENSEMBLE_VIEWS,\n                video_meta=')
    text = replace_once(text, '                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,',
                        '                use_offset=True if self.mode == "val" else self.cfg.DATA.USE_OFFSET_SAMPLING,')
    old = ('                if self.mode not in ["test"] and i_try > self._num_retries // 2:\n'
           '                    # let\'s try another one\n'
           '                    index = random.randint(0, len(self._path_to_videos) - 1)\n'
           '                continue')
    if text.count(old) != 2:
        raise ValueError("Decode failure patch context changed")
    text = text.replace(old, '                raise RuntimeError("Cannot decode prepared clip: " + self._path_to_videos[index])')
    changes[path] = text
    path = "slowfast/models/head_helper.py"
    text = (VENDOR / path).read_text()
    text = replace_once(text, 'from detectron2.layers import ROIAlign\n', '')
    text = replace_once(text, '        super(ResNetRoIHead, self).__init__()',
                        '        # Detection-only dependency; MViT classification never instantiates this head.\n'
                        '        from detectron2.layers import ROIAlign\n'
                        '        super(ResNetRoIHead, self).__init__()')
    changes[path] = text
    path = "tools/train_net.py"
    text = (VENDOR / path).read_text()
    text = replace_once(text, 'import warnings\n', 'import warnings\nimport random\n')
    text = replace_once(text, '    np.random.seed(cfg.RNG_SEED)  # same seed under the same script',
                        '    random.seed(cfg.RNG_SEED)\n    np.random.seed(cfg.RNG_SEED)  # same seed under the same script')
    text = replace_once(text, '    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):\n',
                        '    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):\n'
                        '        # Epoch reseeding makes an epoch-boundary resume reproducible.\n'
                        '        random.seed(cfg.RNG_SEED + cur_epoch)\n'
                        '        np.random.seed(cfg.RNG_SEED + cur_epoch)\n'
                        '        torch.manual_seed(cfg.RNG_SEED + cur_epoch)\n'
                        '        torch.cuda.manual_seed_all(cfg.RNG_SEED + cur_epoch)\n')
    text = replace_once(text, '    val_meter.reset()\n\n    return mAP',
                        '    if cfg.TRAIN.DATASET == "aicity" and cfg.NUM_GPUS == 1:\n'
                        '        from project_hooks import record_validation\n'
                        '        record_validation(model, val_meter, cur_epoch, cfg, writer)\n'
                        '    val_meter.reset()\n\n    return mAP')
    changes[path] = text
    path = "slowfast/datasets/multigrid_helper.py"
    changes[path] = replace_once((VENDOR / path).read_text(),
                                "if TORCH_MAJOR >= 1 and TORCH_MINOR >= 8:",
                                "if (TORCH_MAJOR, TORCH_MINOR) >= (1, 8):")
    path = "slowfast/utils/checkpoint.py"
    changes[path] = replace_once((VENDOR / path).read_text(),
                                'torch.load(f, map_location="cpu")',
                                'torch.load(f, map_location="cpu", weights_only=True)')
    path = "slowfast/models/attention.py"
    text = (VENDOR / path).read_text()
    text = replace_once(text, 'from slowfast.models.common import DropPath, Mlp\n',
                        'from slowfast.models.common import DropPath, Mlp\n'
                        'from slowfast.models.memory_attention import pooled_scaled_attention\n')
    text = replace_once(text, '        attn = (q @ k.transpose(-2, -1)) * self.scale\n'
                        "        # attn: [B, num_head, new_thw, new_thw']\n"
                        '        attn = attn.softmax(dim=-1)',
                        '        attended = pooled_scaled_attention(q, k, v, self.scale)')
    changes[path] = replace_once(text, '        x = (attn @ v).transpose(1, 2).reshape(B, N, C)',
                                '        x = attended.transpose(1, 2).reshape(B, N, C)')
    changes["slowfast/models/memory_attention.py"] = (ROOT / "patches/memory_attention.py.txt").read_text()
    diff = []
    for path, new in changes.items():
        target = VENDOR / path
        if path in lock["upstream_files"]:
            if sha256(target) != lock["upstream_files"][path]:
                raise ValueError(f"Unrecognized upstream file: {path}")
        elif target.exists():
            raise ValueError(f"Unexpected existing added file: {path}")
        before = target.read_text() if target.exists() else ""
        diff.extend(difflib.unified_diff(before.splitlines(True), new.splitlines(True),
                                         fromfile="a/" + path if before else "/dev/null", tofile="b/" + path))
    for path, new in changes.items():
        (VENDOR / path).write_text(new, encoding="utf-8", newline="\n")
    lock["adapted_files"] = {path: sha256(VENDOR / path) for path in changes}
    write_json(ROOT / "source_lock.json", lock)
    (ROOT / "patches").mkdir(exist_ok=True)
    (ROOT / "patches/prepared-data.patch").write_text("".join(diff), encoding="utf-8", newline="\n")
    print("Applied and fingerprinted", len(changes), "source adaptations")


if __name__ == "__main__":
    main()
