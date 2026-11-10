"""Combine verified A1 parts without duplicating videos or changing held-out drivers."""
from pathlib import Path
import sys,json,csv,hashlib,copy,shutil,collections,os,subprocess
ROOT=Path(r'F:\M2DAR_preprocessed');PARTS=[Path('F:/')/name for name in ['A1_preprocessed','A1_2_preprocessed','A1_3_preprocessed','A1_4_preprocessed']]
sys.path.insert(0,r'F:\A1_analysis\tools')
import numpy as np

def run():
    ROOT.mkdir(exist_ok=True);(ROOT/'m2dar').mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parent/'MVITV2_S_16x4_448_prepared.yaml',ROOT/'m2dar'/'MVITV2_S_16x4_448_prepared.yaml')
    shutil.copy2(Path(__file__).resolve().parent/'model_config_validation.json',ROOT/'m2dar'/'model_config_validation.json')
    allgroups=[];packages=[];seen_drivers=set();val_recordings=[];all_issues=[];all_corrections=[]
    class_schema=json.loads((PARTS[0]/'classes.json').read_text())
    for part in PARTS:
        ready=json.loads((part/'READY.json').read_text());smoke=json.loads((part/'training_smoke_test.json').read_text())
        assert ready['status']=='training_ready_subset' and not smoke['driver_leakage'] and not smoke['model_training_executed']
        assert class_schema==json.loads((part/'classes.json').read_text())
        rows=[json.loads(x) for x in (part/'training_manifest.jsonl').read_text().splitlines() if x.strip()]
        assert len(rows)==ready['prepared_triplets']
        drivers=set(ready['train_drivers'])|set(ready['val_drivers']);assert not drivers&seen_drivers;seen_drivers|=drivers
        for g0 in rows:
            g=copy.deepcopy(g0);g['source_package']=part.name
            for v,spec in g['views'].items():
                source=part/spec['relative_output_path'];assert source.is_file()
                spec['relative_output_path']=Path(os.path.relpath(source,ROOT)).as_posix()
            allgroups.append(g)
        audit=json.loads((part/'annotation_audit.json').read_text())
        if (part/'annotation_corrections.json').exists():all_corrections.extend(dict(package=part.name,**r) for r in json.loads((part/'annotation_corrections.json').read_text()))
        for p,sha in audit['source_annotation_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
        packages.append(dict(package=part.name,path=str(part),ready_sha256=hashlib.sha256((part/'READY.json').read_bytes()).hexdigest(),
            source_videos=audit['video_files'],triplets=len(rows),view_clips=3*len(rows),prepared_GB=ready['prepared_GB'],
            train_drivers=ready['train_drivers'],val_drivers=ready['val_drivers'],excluded_annotation_rows=audit['excluded_rows'],
            quarantined_triplets=ready['quarantined_triplets']))
        val_recordings.extend(json.loads((part/'validation_untrimmed_recordings.json').read_text()))
        all_issues.extend(dict(package=part.name,**x) for x in json.loads((part/'annotation_issues.json').read_text()))
    assert len({g['group_id'] for g in allgroups})==len(allgroups)
    train=[g for g in allgroups if g['split']=='train'];val=[g for g in allgroups if g['split']=='val']
    train_drivers={g['driver'] for g in train};val_drivers={g['driver'] for g in val};assert not train_drivers&val_drivers
    a2_sources=json.loads(Path(r'F:\A2_preprocessed\source_videos.json').read_text());a2_drivers={r['driver'] for r in a2_sources}
    assert not (train_drivers|val_drivers)&a2_drivers
    assert set(g['label'] for g in train)==set(range(16))
    active=[]
    for name,groups in [('train',train),('val',val)]:
        (ROOT/f'{name}_groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in groups))
        entries=[]
        for g in groups:
            assert g['validation']['status']=='ready'
            for v in ['dashboard','rearview','right_side_window']:
                path=g['views'][v]['relative_output_path'];resolved=(ROOT/path).resolve();assert resolved.is_file()
                entries.append(f'{path} {g["label"]}\n');active.append(str(resolved))
        (ROOT/'m2dar'/f'{name}.csv').write_text(''.join(entries))
    assert len(active)==len(set(active))==3*len(allgroups)
    (ROOT/'training_manifest.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in allgroups))
    (ROOT/'classes.json').write_text(json.dumps(class_schema,indent=2))
    (ROOT/'splits.json').write_text(json.dumps(dict(unit='driver',seed=42,selection='Two held-out drivers per downloaded A1 part, preserving prior A1 split.',
        train=sorted(train_drivers),val=sorted(val_drivers),A2_excluded_drivers=sorted(a2_drivers),test=[]),indent=2))
    (ROOT/'source_packages.json').write_text(json.dumps(packages,indent=2))
    (ROOT/'validation_untrimmed_recordings.json').write_text(json.dumps(val_recordings,indent=2))
    (ROOT/'annotation_issues.json').write_text(json.dumps(all_issues,indent=2))
    (ROOT/'annotation_corrections.json').write_text(json.dumps(all_corrections,indent=2))
    shutil.copy2(PARTS[0]/'training_dataset.py',ROOT/'training_dataset.py')
    sys.path.insert(0,str(ROOT));from training_dataset import SynchronizedActionDataset
    checks=[]
    for split in ['train','val']:
        ds=SynchronizedActionDataset(ROOT,split)
        indices={0,len(ds)-1}
        for part in PARTS:
            indices.add(next(i for i,g in enumerate(ds.groups) if g['source_package']==part.name))
        for i in sorted(indices):
            item=ds[i];a=item['video'];assert a.shape==(3,3,16,448,448) and a.dtype==np.float32 and np.isfinite(a).all()
            checks.append(dict(split=split,group_id=item['group_id'],passed=True))
    summary=dict(status='training_ready_subset_pending_final_verification',source_parts=[p.name for p in PARTS],source_videos=sum(p['source_videos'] for p in packages),
        source_drivers=len(seen_drivers),prepared_triplets=len(allgroups),prepared_view_clips=3*len(allgroups),
        train_triplets=len(train),val_triplets=len(val),train_csv_entries=3*len(train),val_csv_entries=3*len(val),
        train_drivers=sorted(train_drivers),val_drivers=sorted(val_drivers),A2_drivers_excluded=sorted(a2_drivers),
        prepared_GB=sum(p['prepared_GB'] for p in packages),output_size=[960,540],output_fps=30,classes=16,
        train_class_groups=dict(collections.Counter(g['label'] for g in train)),val_class_groups=dict(collections.Counter(g['label'] for g in val)),
        source_reference_hours=sum(g['frame_count']/30 for g in allgroups)/3600,
        shared_video_files=True,duplicated_media_bytes=0,model_training_executed=False,
        limitation='Conservative motion-aligned subset, not hardware-clock ground truth; coverage omissions and class imbalance are recorded per package.')
    rows=[dict(split=s,class_id=c,triplets=sum(g['label']==c for g in gg),view_clips=3*sum(g['label']==c for g in gg)) for s,gg in [('train',train),('val',val)] for c in range(16)]
    with (ROOT/'class_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'combined_smoke_test.json').write_text(json.dumps(dict(manifest_paths_exist=True,no_duplicate_media_entries=True,driver_leakage=False,
        A2_in_training=False,source_label_hashes_unchanged=True,model_input_checks=checks,model_training_executed=False),indent=2))
    readme=f'''# Combined M2DAR training inputs

Ready: **{len(allgroups)} three-camera groups / {3*len(allgroups)} MP4 clips** from the 40 locally downloaded A1 drivers (240 source videos). Training has {len(train)} groups / {3*len(train)} per-view clips from {len(train_drivers)} drivers. Internal validation has {len(val)} groups / {3*len(val)} clips from {len(val_drivers)} different drivers. This covers the four downloaded A1 parts, not all 69 drivers in the published release.

## Start classification training

Use `m2dar/train.csv` and `m2dar/val.csv` with the existing M2DAR Aicity loader:

```text
DATA.PATH_TO_DATA_DIR F:/M2DAR_preprocessed/m2dar
DATA.PATH_PREFIX F:/M2DAR_preprocessed
DATA.PATH_LABEL_SEPARATOR " "
DATA.NUM_FRAMES 16
DATA.SAMPLING_RATE 4
MODEL.NUM_CLASSES 16
```

The baseline config uses 448-pixel model input and RGB normalization mean 0.45 / std 0.225. All exported clips retain the full camera field of view at 960x540, exactly 30 fps. Each CSV row is a relative path followed by a zero-based class ID. The files live in sibling `A1_preprocessed`, `A1_2_preprocessed`, `A1_3_preprocessed`, and `A1_4_preprocessed` directories: this combined folder references them without copying videos. Keep those siblings together when moving the package and update the two root paths above. Media size across all four is {summary['prepared_GB']:.3f} GB.

`m2dar/MVITV2_S_16x4_448_prepared.yaml` is a complete copy of the pinned repository's MViTv2-S training configuration with the prepared data paths, an explicit space separator and a separate output directory. Its model/optimizer settings are retained. Launch it from the M2DAR repository root after installing the model environment and supplying a compatible initializer. This preparation checked data loading, not execution of this training configuration.

**Checkpoint provenance matters for the driver split.** Start from a verified generic action-pretrained initializer without exposure to the held-out SynDD2 drivers. The repository's training initializer is named `MViTv2_S_16x4_pretrained.pyth`; verify the actual checkpoint's origin and architecture before loading it. Do not initialize this experiment from `MViTv2_S_16x4_aicity_200ep.pyth` or another checkpoint trained on all original A1 drivers: the original paper trained on all A1, including drivers held out here. That would invalidate a driver-independent validation claim. `AUTO_RESUME` is false and the example output directory is separate to avoid accidental reuse of old runs.

The local M2DAR paper describes MViTv2-B pretrained on Kinetics-700, while the released YAML and checkpoint filenames use MViTv2-S. This example explicitly follows the released S configuration. The 16x4 data preparation works for either compatible architecture, but architecture, initializer provenance and head dimensions must be recorded and checked in the model experiment; the checkpoint bytes were not downloaded or audited during preprocessing. Relevant sources: [released training config](https://github.com/PurdueDigitalTwin/M2DAR/blob/8fc6a3f76fc47207c4fa072dff7259f0debb5472/configs/MVITV2_S_16x4_448.yaml), [training versus inference checkpoints](https://github.com/PurdueDigitalTwin/M2DAR/blob/8fc6a3f76fc47207c4fa072dff7259f0debb5472/README.md), and the implementation-details section of `F:/M2DAR_base.pdf`.

The driver split preserves the earlier A1 split and uses two held-out drivers per additional part, with seed 42. Every camera and recording of a driver stays in the same partition. All 16 classes are represented in training; `class_counts.csv` records coverage in both partitions. Samples from the three cameras of one action are correlated views, not independent action occurrences. Use the identical prepared split for M2DAR and the proposed model in all comparative experiments. The published 2023 M2DAR score is not directly comparable with this 2024 subset and split.

## Synchronized multi-view loader

```python
from training_dataset import SynchronizedActionDataset
dataset = SynchronizedActionDataset(r"F:\\M2DAR_preprocessed", split="train")
sample = dataset[0]
print(sample['video'].shape)  # (3, 3, 16, 448, 448)
```

`train_groups.jsonl` and `val_groups.jsonl` preserve camera grouping, label provenance and timing mappings. The optional loader uses identical temporal indices across all views. It resizes the full image to 448 square; M2DAR's native training loader uses its configured spatial transforms. Record this transform choice in experiments. Short clips retain only observed footage, and bounded model sampling may repeat the last valid frame. Call `set_epoch(epoch)` for varying training windows. `as_torch=True` returns tensors when PyTorch is installed in your model environment.

## Synchronization and exclusions

All new sources were fully decoded. Alignment uses dense PTS-based visual motion, dashboard reference time, and piecewise camera offsets. Only supported shared intervals with the same supplied action class across the three cameras were exported; a 0.25-second label margin is applied. Every retained export was independently fully decoded, and strong residual motion mismatches were quarantined. Short/static checks can be inconclusive, so this is estimated synchronization rather than a hardware-clock calibration. Missing labels, invalid annotations and missing footage were not guessed. `annotation_corrections.json` records the inspected driver-47158 camera-prefix repair using the supplied Camera View column; class IDs and times are unchanged. Consult each source package's `annotation_issues.json`, `alignment_quarantine.json`, `clip_plan_omissions.json`, and `quarantined_exports.json` for exclusions. The original source videos and labels are unchanged.

## A2 and temporal localization

**A2 is not part of either labelled CSV.** Its 15 drivers are separate from these 40 A1 drivers. `F:/A2_preprocessed/README.md` describes the unlabelled full-recording inputs, aligned inference chunks, original-time mappings and model input loader. No A2 pseudo-labels or action labels were created, and no A2 accuracy can be calculated locally without ground truth. Do not use A2 for supervised or semi-supervised training; see the [official 2024 FAQ](https://www.aicitychallenge.org/2024-faqs/).

The internal A1 validation CSV supports clip classification. Full temporal-localization validation requires untrimmed recordings and annotations. `validation_untrimmed_recordings.json` preserves all held-out recording references, motion mappings and original annotation statuses; ignore annotation rows marked excluded. The original 2023 M2DAR thresholds and one-event-per-class selection are not validated for the expanded 2024 data. Tune experimental choices using the held-out A1 drivers, then keep A2 for final inference/evaluation.

No neural model was trained and no model prediction scores were computed during this preprocessing. Train using your M2DAR model environment and pretrained weights. The CPU preprocessing environment is recorded separately in the source packages; it should not replace M2DAR's pinned model dependencies.

## Reproduce or inspect

- `source_packages.json` identifies the four ready packages and their completion hashes.
- `READY.json`, `combined_smoke_test.json`, and each package's full validation files record the checks.
- `F:/M2DAR_preprocessing/run_extension.py` is the resumable extension entry point. It reuses unchanged feature and clip caches. Do not run multiple writers on the same package.
- Source dataset roles follow the [2024 Track 3 guidance](https://www.aicitychallenge.org/2024-faqs/). Baseline interface: [M2DAR](https://github.com/PurdueDigitalTwin/M2DAR/tree/8fc6a3f76fc47207c4fa072dff7259f0debb5472).
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    (ROOT/'READY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    run()
    for name in ['final_verify.py','make_review_surveys.py']:
        subprocess.run([sys.executable,str(Path(__file__).resolve().parent/name)],check=True)
