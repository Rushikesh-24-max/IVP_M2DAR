"""Document the verified dataset and segregate generated clips not in its manifest."""
from pathlib import Path
import json,sys,hashlib,shutil,collections,platform,csv
OUT=Path(__file__).resolve().parent
sys.path.insert(0,r'F:\A1_analysis\tools')
import av,numpy as np,PIL

def canonical_validation_metadata(obj):
    if isinstance(obj,dict):
        if obj.get('status') in ('motion_check_inconclusive','too_short_for_independent_motion_check'):
            if 'residual_frames' in obj:obj['residual_frames']=None
            if 'residual_s' in obj:obj['residual_s']=None
        for value in obj.values():canonical_validation_metadata(value)
    elif isinstance(obj,list):
        for value in obj:canonical_validation_metadata(value)
    return obj

def run():
    # An argmax on an uninformative signal is not a measured residual lag.
    for name in ['validation_cache.json','export_validation.json','quarantined_exports.json']:
        p=OUT/name
        if p.exists():p.write_text(json.dumps(canonical_validation_metadata(json.loads(p.read_text())),indent=2))
    for name in ['training_manifest.jsonl','train_groups.jsonl','val_groups.jsonl']:
        p=OUT/name
        rows=[canonical_validation_metadata(json.loads(x)) for x in p.read_text().splitlines() if x]
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    ready=json.loads((OUT/'READY.json').read_text());audit=json.loads((OUT/'annotation_audit.json').read_text())
    groups=[json.loads(x) for x in (OUT/'training_manifest.jsonl').read_text().splitlines() if x]
    active={str((OUT/g['views'][v]['relative_output_path']).resolve()) for g in groups for v in g['views']}
    clips=(OUT/'clips').resolve();quarantine=(OUT/'quarantine'/'unpublished_clips').resolve();moved=[]
    for p in sorted(clips.rglob('*.mp4')):
        source=p.resolve()
        if str(source) in active:continue
        if not source.is_relative_to(clips):raise ValueError('Unexpected source path')
        dest=quarantine/source.relative_to(clips)
        if not dest.resolve().is_relative_to(quarantine):raise ValueError('Unexpected destination path')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():dest=dest.with_name(dest.stem+'_extra'+str(source.stat().st_mtime_ns)+dest.suffix)
        old_meta=source.with_suffix('.json');source.replace(dest)
        if old_meta.exists():old_meta.replace(dest.with_suffix('.json'))
        moved.append(dict(source=str(source),destination=str(dest)))
    journal=OUT/'unpublished_clip_moves.json';previous_moves=json.loads(journal.read_text()) if journal.exists() else []
    journal.write_text(json.dumps(previous_moves+moved,indent=2))
    metadata=OUT/'metadata';annotations=metadata/'source_annotations';annotations.mkdir(parents=True,exist_ok=True)
    for p in audit['source_annotation_sha256']:
        shutil.copy2(p,annotations/Path(p).name)
    for name in ['README.txt','Distracted_Activity_Class_definition.txt']:
        shutil.copy2(Path(r'F:\labels&instructions')/name,metadata/name)
    env=dict(python=platform.python_version(),av=av.__version__,numpy=np.__version__,Pillow=PIL.__version__,platform=platform.platform())
    (OUT/'environment_preprocessing.json').write_text(json.dumps(env,indent=2))
    (OUT/'requirements-preprocessing.txt').write_text(f'av=={av.__version__}\nnumpy=={np.__version__}\nPillow=={PIL.__version__}\n')
    mappings=json.loads((OUT/'alignment_estimates.json').read_text());val_drivers=set(ready['val_drivers'])
    val_recordings=[dict(g,reference_annotations=[r for r in json.loads((OUT/'annotations_normalized.json').read_text()) if r['driver']==g['driver'] and r['session']==g['session'] and r['view']=='dashboard']) for g in mappings['groups'] if g['driver'] in val_drivers]
    (OUT/'validation_untrimmed_recordings.json').write_text(json.dumps(val_recordings,indent=2))
    label_checks=json.loads((OUT/'annotation_issues.json').read_text());omissions=json.loads((OUT/'clip_plan_omissions.json').read_text())
    references={(g['driver'],g['session'],g['views']['dashboard']['source_annotation_row']) for g in groups}
    prepared_bytes=sum((OUT/g['views'][v]['relative_output_path']).stat().st_size for g in groups for v in g['views'])
    ready.update(prepared_GB=prepared_bytes/1e9,retained_reference_action_annotations=len(references),short_triplets=sum(g.get('temporal_padding_required',False) for g in groups),
        valid_source_label_rows=audit['valid_rows'],excluded_source_label_rows=audit['excluded_rows'],annotation_rows=audit['annotation_rows'],
        omitted_reference_annotations=len(omissions),source_videos_unchanged=True,source_annotation_hashes_unchanged=True)
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2))
    config=dict(scope='A1_extracted only',reference_view='dashboard',source_time_mapping='source_time = reference_time + offset',
        output_fps=30,output_size=[960,540],field_of_view='full source image, aspect ratio preserved',encoder='libx264',crf=20,preset='veryfast',
        label_edge_margin_seconds=.25,minimum_observed_clip_frames=16,model_input_frames=16,model_sampling_rate=4,model_crop_size=448,
        model_normalization=dict(mean=[.45]*3,std=[.225]*3,rgb_scale=255),
        alignment=dict(pixel_proxy=[128,72],motion_window_seconds=15,window_step_seconds=4,search_seconds=1.5,
                       minimum_correlation=.65,minimum_peak_margin=.06,fit_residual_frames=2,post_export_residual_limit_frames=3,
                       interpretation='motion estimates, not hardware-clock ground truth'),
        script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
    (OUT/'processing_config.json').write_text(json.dumps(config,indent=2))
    rows=[]
    for split in ['train','val']:
        for label in range(16):rows.append(dict(split=split,class_id=label,triplets=sum(g['split']==split and g['label']==label for g in groups),view_clips=3*sum(g['split']==split and g['label']==label for g in groups)))
    with (OUT/'class_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    readme=f'''# Prepared A1 training data

**Ready: {ready['prepared_triplets']} aligned three-camera clip triplets ({ready['prepared_view_clips']} MP4s), {ready['prepared_GB']:.3f} GB.**

Training: {ready['train_triplets']} triplets, {ready['train_triplets']*3} per-view clips, {len(ready['train_drivers'])} drivers. Validation: {ready['val_triplets']} triplets, {ready['val_triplets']*3} per-view clips, {len(ready['val_drivers'])} drivers. All 16 classes are present in training. See `class_counts.csv` for actual class coverage.

The included clips are a conservative subset of A1. All 60 source videos were processed. The exported subset covers {ready['retained_reference_action_annotations']} distinct source reference annotations and {ready['source_reference_hours']*60:.2f} minutes of common action time. Fragment counts are not independent driver/event counts. Original videos and label files are unchanged.

## Start with M2DAR

Use the existing M2DAR Aicity data loader with:

- `DATA.PATH_TO_DATA_DIR`: `F:/A1_preprocessed/m2dar`
- `DATA.PATH_PREFIX`: `F:/A1_preprocessed`
- `DATA.PATH_LABEL_SEPARATOR`: one space
- `DATA.NUM_FRAMES`: 16; `DATA.SAMPLING_RATE`: 4; target FPS: 30
- Model spatial input: 448; RGB normalization mean 0.45 / std 0.225; 16 classes

`m2dar/train.csv` and `m2dar/val.csv` contain relative video paths followed by zero-based class IDs. Use M2DAR's own training environment and pretrained weights. This package prepares data; it does not include model weights or a completed training run. When copying the data to another machine, change the two path overrides to the copied package root.

## Synchronized multi-view models

`train_groups.jsonl` and `val_groups.jsonl` retain camera grouping, class, driver, source intervals and timing offsets. The cameras are ordered dashboard, rearview, right-side window. An optional loader is supplied:

```python
from training_dataset import SynchronizedActionDataset
data = SynchronizedActionDataset(r"F:\\A1_preprocessed", split="train")
sample = data[0]
print(sample["video"].shape, sample["label"])
# video: (3 views, 3 RGB channels, 16 frames, 448, 448), float32
```

It uses identical temporal sample indices for all views. Set `as_torch=True` in an environment with PyTorch if tensors are desired. Call `set_epoch(epoch)` when training to vary the deterministic training window between epochs. The helper resizes the full image to 448 square; M2DAR's native loader uses its configured spatial transforms. Treat these as explicitly different training transforms, not identical experiments.

## Processing choices

- Each clip is 960x540 H.264 at exactly 30 fps, with its timeline reset to zero. The complete camera field of view is retained. No audio is added.
- Source frames are sampled by presentation timestamp, using dashboard time as reference. Camera mappings are piecewise where the motion evidence supports a timing change.
- Clips use the common intersection of matching class annotations in all three views, with a 0.25-second label-edge margin.
- {ready['short_triplets']} triplets are shorter than the baseline's 64-frame support. Their MP4s contain only observed footage. The model sampler can repeat the valid boundary frame, matching M2DAR's bounded-index behavior; `temporal_padding_required` flags these cases.
- The split is by driver, seed 42. Training drivers: {', '.join(ready['train_drivers'])}. Validation drivers: {', '.join(ready['val_drivers'])}. Both recordings and all cameras of each driver stay in one partition. A2 and B were not used.

## Exclusions and synchronization limits

Of {audit['annotation_rows']} local annotation rows, {audit['excluded_rows']} were excluded: eight outside available footage, three zero-duration rows, and three copies of a long conflicting annotation. The `16700/7` right-side video ends at 268 seconds. No missing camera footage or missing class labels were invented. Uncertain timing transitions, sustained disagreements and unsupported intervals were kept outside the prepared subset.

After export, every included MP4 was fully decoded and checked for equal triplet frame counts, exact 30-fps presentation times, duration and corrupt-frame flags. Informative post-export motion checks with residual lag over three frames led to triplet quarantine. {ready['quarantined_triplets']} planned triplets were quarantined at that check.

Alignment is estimated from visual motion, not a calibrated shared hardware clock. Static/short/occluded intervals can make an independent motion check inconclusive; these retain their supported surrounding time mapping and passed media checks. No claim of universal frame-level ground-truth accuracy is made. Review `preview/`, `export_validation.json` and the per-group validation fields before final experiments.

The original burned-in camera clocks are preserved and can display different clock readings. They were not calibrated as a common clock. Judge the previews using their added aligned-time counter and corresponding body movements, rather than matching the original clock digits.

`m2dar/val.csv` supports held-out-driver classification validation on prepared clips. Temporal-localization evaluation on long recordings additionally needs the original untrimmed videos and timeline-aware labels. `validation_untrimmed_recordings.json` provides the held-out recording references and mappings; it is not a full challenge test result.

## Files to use

- `clips/`: only clips listed in the ready training/validation manifests
- `m2dar/train.csv`, `m2dar/val.csv`: baseline per-view lists
- `train_groups.jsonl`, `val_groups.jsonl`: synchronized grouping
- `classes.json`, `splits.json`, `class_counts.csv`: labels and split definitions
- `READY.json`, `training_smoke_test.json`: completion and input checks
- `preview/synchronized_example.mp4`: three-camera playback example
- `metadata/source_annotations/`: unchanged snapshots of the ten used CSVs

`annotation_issues.json`, `alignment_quarantine.json`, `clip_plan_omissions.json` and `quarantined_exports.json` explain exclusions. Generated clips not in the final ready set are segregated under `quarantine/`; do not scan that directory into training. Intermediate motion evidence, scripts and source links remain for reproducibility.

## Re-run preprocessing

The current preprocessing environment is recorded in `environment_preprocessing.json`; its dependencies are in `requirements-preprocessing.txt`. Keep these separate from an older M2DAR model environment. With the original F: inputs present:

```text
python run_pipeline.py
```

Use `python run_pipeline.py --cpu` to avoid the Windows D3D11 decoding path. Existing matching motion caches and clips are reused. Re-running is an explicit local data-processing job; it does not launch model training. Do not run multiple writers concurrently on the same output directory.

On this computer, the already available bundled interpreter can also be invoked from PowerShell:

```powershell
& "C:\\Users\\anil\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" "F:\\A1_preprocessed\\run_pipeline.py"
```

Relevant baseline contracts: [M2DAR Aicity CSV reader](https://github.com/PurdueDigitalTwin/M2DAR/blob/8fc6a3f76fc47207c4fa072dff7259f0debb5472/slowfast/datasets/aicity.py) and [M2DAR temporal sampling and bounded indices](https://github.com/PurdueDigitalTwin/M2DAR/blob/8fc6a3f76fc47207c4fa072dff7259f0debb5472/slowfast/datasets/decoder.py). The local original annotation CSVs remain the label authority.
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps(ready,indent=2));print('Generated unpublished clips segregated:',len(moved))

if __name__=='__main__':run()
