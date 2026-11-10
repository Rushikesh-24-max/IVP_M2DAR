"""CPU-only protocol/regression tests; no neural training or model downloads."""
import ast
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import VENDOR, a1_relative_path, apply_gpu_memory_limit, require_shapes, sha256, source_integrity, verify_bundle, write_json
from metrics import classification_metrics
from prepare_a2 import original_relative
from prepare_initializer import audit_provenance
from restore_a2_times import restore
from train import validate_smoke
from checkpoint_storage import retain_recent_checkpoints


class PathsAndIntegrity(unittest.TestCase):
    def test_portable_a1_path(self):
        self.assertEqual(a1_relative_path("../A1_2_preprocessed/clips/100/5/a.mp4", "A1_2_preprocessed"),
                         "A1_2_preprocessed/clips/100/5/a.mp4")

    def test_reject_path_escape_and_a2_training(self):
        for value, package in [("../A2_preprocessed/clips/a.mp4", "A2_preprocessed"),
                               ("../A1_preprocessed/../../a.mp4", "A1_preprocessed"),
                               ("F:/a.mp4", "A1_preprocessed")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                a1_relative_path(value, package)

    def test_original_a2_windows_relocation(self):
        self.assertEqual(original_relative(r"F:\A2\SET-A2\user_id_1\Dashboard.MP4"),
                         "A2/SET-A2/user_id_1/Dashboard.MP4")
        with self.assertRaises(ValueError):
            original_relative(r"F:\A1_extracted\file.mp4")

    def test_bundle_tampering_detected(self):
        with tempfile.TemporaryDirectory(prefix="bundle with spaces ") as temp:
            root = Path(temp)
            (root / "train.csv").write_text("video.mp4\t1\n")
            write_json(root / "dataset.json", {"artifacts": {"train.csv": sha256(root / "train.csv")}})
            verify_bundle(root, check_files=False)
            (root / "train.csv").write_text("video.mp4\t2\n")
            with self.assertRaises(ValueError):
                verify_bundle(root, check_files=False)

    def test_source_lock(self):
        self.assertEqual(len(source_integrity()), 64)

    def test_training_requires_matching_gpu_smoke(self):
        from common import application_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "baseline.yaml").write_text("test fixture")
            (root / "init.bin").write_bytes(b"test fixture")
            manifest = {"dataset_fingerprint": "fixture"}
            report = dict(dataset_fingerprint="fixture", config_sha256=sha256(root / "baseline.yaml"),
                          source_lock_sha256=source_integrity(), application_sha256=application_hash(),
                          checkpoint_sha256=sha256(root / "init.bin"), gpu_smoke_passed=True)
            validate_smoke(report, manifest, root, root / "init.bin")
            limited = dict(report, gpu_memory_fraction=.2)
            validate_smoke(limited, manifest, root, root / "init.bin", .2)
            with self.assertRaises(ValueError):
                validate_smoke(limited, manifest, root, root / "init.bin")
            for key, value in (("gpu_smoke_passed", False), ("checkpoint_sha256", "wrong"),
                               ("dataset_fingerprint", "other-split"), ("config_sha256", "changed")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_smoke(dict(report, **{key: value}), manifest, root, root / "init.bin")


class GpuMemoryBudget(unittest.TestCase):
    def test_budget_applied_to_logical_gpu_zero(self):
        from unittest.mock import patch
        setter = Mock()
        fake = NS(cuda=NS(mem_get_info=lambda device: (7*1024**3, 24*1024**3),
                         set_per_process_memory_fraction=setter))
        with patch.dict(sys.modules, {"torch": fake}):
            budget = apply_gpu_memory_limit(.2)
        setter.assert_called_once_with(.2, device=0)
        self.assertEqual(budget["allocator_limit_bytes"], int(24*1024**3*.2))

    def test_low_free_memory_stops_before_setting_budget(self):
        from unittest.mock import patch
        setter = Mock()
        fake = NS(cuda=NS(mem_get_info=lambda device: (1024**3, 24*1024**3),
                         set_per_process_memory_fraction=setter))
        with patch.dict(sys.modules, {"torch": fake}), self.assertRaises(RuntimeError):
            apply_gpu_memory_limit(.2)
        setter.assert_not_called()

    def test_invalid_budgets_rejected(self):
        for value in (0, -1, 1.1, float("nan"), True, "0.2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                apply_gpu_memory_limit(value)
        self.assertIsNone(apply_gpu_memory_limit(None))


class CheckpointRetention(unittest.TestCase):
    def create_run(self, root, keep=2):
        write_json(root / "run.json", dict(run_id="test-fixture", keep_last_checkpoints=keep))
        (root / "checkpoints").mkdir()
        for epoch in (10, 20, 30):
            (root / "checkpoints" / f"checkpoint_epoch_{epoch:05d}.pyth").write_bytes(b"synthetic fixture")

    def test_keep_two_and_preserve_best_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_run(root)
            (root / "best.pyth").write_bytes(b"best fixture")
            (root / "checkpoints" / "notes.txt").write_text("keep")
            (root / "checkpoints" / "checkpoint_epoch_00040.pyth.tmp").write_bytes(b"incomplete")
            removed = retain_recent_checkpoints(root, 30)
            self.assertEqual([r["name"] for r in removed], ["checkpoint_epoch_00010.pyth"])
            self.assertTrue((root / "checkpoints/checkpoint_epoch_00020.pyth").exists())
            self.assertTrue((root / "checkpoints/checkpoint_epoch_00030.pyth").exists())
            self.assertTrue((root / "checkpoints/notes.txt").exists())
            self.assertTrue((root / "checkpoints/checkpoint_epoch_00040.pyth.tmp").exists())
            self.assertEqual((root / "best.pyth").read_bytes(), b"best fixture")

    def test_no_pruning_before_new_checkpoint_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_run(root)
            (root / "checkpoints/checkpoint_epoch_00030.pyth").write_bytes(b"")
            self.assertEqual(retain_recent_checkpoints(root, 30), [])
            self.assertTrue((root / "checkpoints/checkpoint_epoch_00010.pyth").exists())

    def test_unconfigured_older_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_run(root)
            write_json(root / "run.json", {"run_id": "older-run"})
            self.assertEqual(retain_recent_checkpoints(root, 30), [])
            self.assertEqual(len(list((root / "checkpoints").glob("*.pyth"))), 3)

    def test_invalid_policy_does_not_remove_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_run(root, keep=0)
            with self.assertRaises(ValueError):
                retain_recent_checkpoints(root, 30)
            self.assertEqual(len(list((root / "checkpoints").glob("*.pyth"))), 3)


class MetricsAndCheckpoints(unittest.TestCase):
    def test_perfect_all_class_metrics(self):
        scores = [[float(i == j) for i in range(16)] for j in range(16)]
        result = classification_metrics(list(range(16)), scores)
        self.assertEqual(result["macro_f1"], 1.)
        self.assertEqual(result["accuracy"], 1.)
        self.assertEqual(result["balanced_accuracy"], 1.)

    def test_macro_f1_is_not_accuracy(self):
        result = classification_metrics([0, 0, 1], [[1., 0.]] * 3, classes=2)
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["macro_f1"], .4)

    def test_nonfinite_and_logits_rejected(self):
        for scores in ([float("nan"), 0.], [2., -1.], [.2, .2]):
            with self.assertRaises(ValueError):
                classification_metrics([0], [scores], classes=2)

    def test_backbone_shape_guard(self):
        expected = {"patch.weight": NS(shape=(96, 3, 3, 7, 7)), "head.projection.weight": NS(shape=(16, 768))}
        generic = {"patch.weight": NS(shape=(96, 3, 3, 7, 7))}
        require_shapes(expected, generic, allow_head=True)
        with self.assertRaises(ValueError):
            require_shapes(expected, generic)
        generic["patch.weight"] = NS(shape=(128, 3, 3, 7, 7))
        with self.assertRaises(ValueError):
            require_shapes(expected, generic, allow_head=True)

    def test_checkpoint_provenance_requires_hash_and_no_target_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.bin"
            path.write_bytes(b"unit test only")
            evidence = dict(kind="generic_action_pretraining", source_url="https://example.org/test",
                            training_datasets=["Test fixture"], evidence="unit fixture", aicity_or_syndd2_exposure=False,
                            sha256=sha256(path))
            audit_provenance(path, evidence)
            evidence["aicity_or_syndd2_exposure"] = True
            with self.assertRaises(ValueError):
                audit_provenance(path, evidence)
            evidence["aicity_or_syndd2_exposure"] = False
            evidence["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                audit_provenance(path, evidence)


class PatchedSampling(unittest.TestCase):
    def test_torch2_uses_supported_integer_type(self):
        tree = ast.parse((VENDOR / "slowfast/datasets/multigrid_helper.py").read_text())
        branch = next(n for n in tree.body if isinstance(n, ast.If))
        for major, minor in ((1, 8), (1, 13), (2, 0), (2, 4), (2, 9)):
            namespace = {"TORCH_MAJOR": major, "TORCH_MINOR": minor}
            exec(compile(ast.Module(body=[branch], type_ignores=[]), "torch_version_branch", "exec"), namespace)
            self.assertIs(namespace["_int_classes"], int)

    def make_dataset(self, mode, decode_failure=False):
        tree = ast.parse((VENDOR / "slowfast/datasets/aicity.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Aicity")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
        frames = Mock()
        frames.permute.return_value = frames
        decode = Mock(return_value=None if decode_failure else frames)
        spatial = Mock(return_value=frames)
        namespace = {"random": random, "logger": Mock(), "container": NS(get_video_container=Mock(return_value=object())),
                     "decoder": NS(decode=decode),
                     "utils": NS(get_random_sampling_rate=lambda a, b: b, tensor_normalize=lambda *a: frames,
                                 spatial_sampling=spatial, pack_pathway_output=lambda *a: [frames])}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "patched_getitem", "exec"), namespace)
        cfg = NS(DATA=NS(TRAIN_JITTER_SCALES=[480, 540], TRAIN_CROP_SIZE=448, TEST_CROP_SIZE=448,
                         SAMPLING_RATE=4, NUM_FRAMES=16, TARGET_FPS=30, DECODING_BACKEND="decord",
                         USE_OFFSET_SAMPLING=False, MEAN=[.45]*3, STD=[.225]*3, RANDOM_FLIP=False, INV_UNIFORM_SAMPLE=False),
                 TEST=NS(NUM_ENSEMBLE_VIEWS=10, NUM_SPATIAL_CROPS=1),
                 MULTIGRID=NS(DEFAULT_S=0, LONG_CYCLE_SAMPLING_RATE=0),
                 DATA_LOADER=NS(ENABLE_MULTI_THREAD_DECODE=False))
        instance = NS(mode=mode, cfg=cfg, use_multi_dataset=False, _num_retries=10, aug=False,
                      _path_to_videos=["clip.mp4"], _labels=[8], _video_meta={0: {}}, _spatial_temporal_idx=[0])
        return lambda: namespace["__getitem__"](instance, 0), decode, spatial

    def test_validation_center_time_and_crop(self):
        call, decode, spatial = self.make_dataset("val")
        for _ in range(2):
            self.assertEqual(call()[1:3], (8, 0))
            self.assertEqual(decode.call_args.args[3:5], (0, 1))
            self.assertTrue(decode.call_args.kwargs["use_offset"])
            self.assertEqual(spatial.call_args.kwargs["spatial_idx"], 1)
            self.assertEqual(spatial.call_args.kwargs["min_scale"], spatial.call_args.kwargs["max_scale"])

    def test_training_keeps_random_sampling(self):
        call, decode, spatial = self.make_dataset("train")
        call()
        self.assertEqual(decode.call_args.args[3], -1)
        self.assertEqual(spatial.call_args.kwargs["spatial_idx"], -1)

    def test_decode_failure_never_substitutes_another_video(self):
        for mode in ("train", "val"):
            call, decode, _ = self.make_dataset(mode, decode_failure=True)
            with self.assertRaisesRegex(RuntimeError, "clip.mp4"):
                call()
            self.assertEqual(decode.call_count, 1)


class TimestampRestoration(unittest.TestCase):
    def test_overlap_merge_clipping_and_unknown_gap(self):
        groups = [dict(group_id="a", video_id=1, frame_count=300, reference_start_s=100.),
                  dict(group_id="b", video_id=1, frame_count=300, reference_start_s=108.),
                  dict(group_id="c", video_id=1, frame_count=300, reference_start_s=130.)]
        result = restore(["a 2 7 12", "b 2 0 4", "c 2 0 3", "a 3 0 0"], groups)
        self.assertEqual([(r["start_s"], r["end_s"]) for r in result], [(107., 112.), (130., 133.)])
        self.assertEqual(result[0]["source_segments"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
