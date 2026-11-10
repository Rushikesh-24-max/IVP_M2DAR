"""Prepare unlabelled A2 for timeline-aware inference, never for training."""
from pathlib import Path
import sys,json,csv,ast,math,hashlib,collections,concurrent.futures,time,shutil,subprocess
ROOT=Path(r'F:\A2_preprocessed');BASE=Path(r'F:\A1_preprocessed')
sys.path.insert(0,str(ROOT));sys.path.insert(0,r'F:\A1_analysis\tools')
VIEWS=['dashboard','rearview','right_side_window'];FPS=30;MAX_FRAMES=900;OVERLAP=64

def load_ids(sources):
    from audit_labels_generic import identify
    by={(r['driver'],r['session'],r['view']):r for r in sources};ids={};records=[]
    for line in Path(r'F:\2024-data_video_ids.csv').read_text(encoding='utf-8-sig').splitlines()[1:]:
        if not line.strip():continue
        number,value=line.split(',',1);names=ast.literal_eval(value);assert len(names)==3
        keys=[identify(name) for name in names]
        assert len({k[:2] for k in keys})==1 and {k[2] for k in keys}==set(VIEWS)
        key=keys[0][:2];assert key not in ids
        ids[key]=int(number)
        records.append(dict(video_id=int(number),driver=key[0],session=key[1],views={k[2]:by[k] for k in keys},original_filenames=names))
    assert len(records)==30 and len(sources)==90 and set(ids)=={(r['driver'],r['session']) for r in sources}
    return ids,records

def union(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]:result[-1][1]=max(b,result[-1][1])
        else:result.append([a,b])
    return result

def plan(sources,ids):
    by={(r['driver'],r['session'],r['view']):r for r in sources}
    mappings=json.loads((ROOT/'alignment_estimates.json').read_text())['groups'];jobs=[];omitted=[]
    assert len(mappings)==30
    for g in mappings:
        d,s=g['driver'],g['session'];vv=g['views'];n=0
        for rear in vv['rearview']['segments']:
            for right in vv['right_side_window']['segments']:
                offsets={'dashboard':0,'rearview':rear['offset_s'],'right_side_window':right['offset_s']}
                start=max(rear['reference_start_s'],right['reference_start_s'],*[by[(d,s,v)].get('first_pts_s',0)-offsets[v] for v in VIEWS])
                end=min(rear['reference_end_s'],right['reference_end_s'],*[by[(d,s,v)]['duration_s']-offsets[v] for v in VIEWS])
                a=math.ceil((start+.05)*FPS-1e-6);b=math.floor((end-.05)*FPS+1e-6)
                if b-a<64:continue
                while a+64<=b:
                    stop=min(a+MAX_FRAMES,b);gid=f'a2_v{ids[(d,s)]:03d}_s{n:03d}';n+=1
                    job=dict(group_id=gid,video_id=ids[(d,s)],driver=d,session=s,label=None,split='inference',
                        fps=FPS,frame_count=stop-a,reference_start_frame=a,reference_end_frame=stop,
                        reference_start_s=a/FPS,reference_end_s=stop/FPS,
                        temporal_padding_required=False,action_ground_truth_available=False,views={})
                    for v in VIEWS:
                        evidence={'dashboard':{},'rearview':rear,'right_side_window':right}[v]
                        job['views'][v]=dict(source_path=by[(d,s,v)]['path'],source_start_s=a/FPS+offsets[v],source_end_s=stop/FPS+offsets[v],offset_s=offsets[v],
                            relative_output_path=f'clips/{d}/{s}/{gid}_{v}.mp4',
                            alignment_anchor_count=evidence.get('accepted_anchor_count'),maximum_supported_anchor_residual_frames=evidence.get('max_anchor_residual_frames',0),
                            alignment_interval_reference_s=[evidence.get('reference_start_s',start),evidence.get('reference_end_s',end)])
                    job['offset_frames']={v:int(round(offsets[v]*FPS)) for v in VIEWS};jobs.append(job)
                    if stop==b:break
                    a=stop-OVERLAP
        if n==0:omitted.append(dict(video_id=ids[(d,s)],driver=d,session=s,reason='no_supported_common_alignment_interval'))
    (ROOT/'clip_plan.json').write_text(json.dumps(dict(schema_version=1,role='unlabelled_inference_only',fps=30,width=960,height=540,
        reference_view='dashboard',max_chunk_frames=MAX_FRAMES,chunk_overlap_frames=OVERLAP,groups=jobs),indent=2))
    (ROOT/'clip_plan_omissions.json').write_text(json.dumps(omitted,indent=2));return jobs

def export(jobs):
    # Copy only already inspected reusable media functions, never training publication.
    for name in ['export_clips.py','validate_exports.py','fit_alignment.py','create_previews.py']:
        if name=='fit_alignment.py' and (ROOT/name).exists():continue
        if name=='create_previews.py':
            (ROOT/name).write_text((BASE/name).read_text().replace('class {g["label"]}','unlabelled'))
        else:shutil.copy2(BASE/name,ROOT/name)
    from export_clips import export_view
    from validate_exports import check_group
    tasks=[(g,v) for g in jobs for v in VIEWS];results=[];started=time.time()
    def one(g,v):
        if shutil.disk_usage(ROOT).free<4_000_000_000:raise RuntimeError('Less than 4 GB free; no new export started')
        return export_view(g,v)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(one,g,v):(g,v) for g,v in tasks}
        for fut in concurrent.futures.as_completed(pending):
            g,v=pending[fut]
            try:r=fut.result()
            except Exception as e:r=dict(group_id=g['group_id'],view=v,error=repr(e))
            results.append(r)
            if len(results)%12==0 or len(results)==len(tasks):
                (ROOT/'export_progress.json').write_text(json.dumps(dict(done=len(results),total=len(tasks),elapsed_s=time.time()-started,
                    errors=[r for r in results if 'error' in r]),indent=2))
                print('A2 EXPORTED',len(results),'/',len(tasks),'elapsed',round(time.time()-started,1),flush=True)
    (ROOT/'export_results.json').write_text(json.dumps(results,indent=2))
    failures=[r for r in results if 'error' in r]
    if failures:raise RuntimeError('A2 export errors: '+str(failures[:3]))
    cache_path=ROOT/'validation_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    checked=[];todo=[]
    for g in jobs:
        old=cache.get(g['group_id']);valid=bool(old and 'error' not in old)
        for v in VIEWS:
            st=(ROOT/g['views'][v]['relative_output_path']).stat();prev=old.get('output_file_stats',{}).get(v,{}) if old else {}
            if prev.get('size_bytes')!=st.st_size or prev.get('mtime_ns')!=st.st_mtime_ns:valid=False
        if valid:checked.append(old)
        else:todo.append(g)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(check_group,g):g for g in todo}
        for fut in concurrent.futures.as_completed(pending):
            g=pending[fut]
            try:r=fut.result()
            except Exception as e:r=dict(group_id=g['group_id'],status='error',error=repr(e))
            checked.append(r);cache[g['group_id']]=r
            if len(checked)%10==0 or len(checked)==len(jobs):
                cache_path.write_text(json.dumps(cache,indent=2));print('A2 CHECKED',len(checked),'/',len(jobs),flush=True)
    cache_path.write_text(json.dumps(cache,indent=2));(ROOT/'export_validation.json').write_text(json.dumps(checked,indent=2))
    if any(r.get('error') for r in checked):raise RuntimeError('An A2 export could not be fully validated')
    by={r['group_id']:r for r in checked}
    ready=[dict(g,validation=by[g['group_id']]) for g in jobs if by[g['group_id']]['status']=='ready']
    bad=[dict(g,validation=by[g['group_id']]) for g in jobs if by[g['group_id']]['status']!='ready']
    (ROOT/'inference_groups.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in ready))
    (ROOT/'quarantined_exports.json').write_text(json.dumps(bad,indent=2))
    active={(ROOT/g['views'][v]['relative_output_path']).resolve() for g in ready for v in VIEWS};moves=[]
    clips=(ROOT/'clips').resolve();quarantine=(ROOT/'quarantine').resolve()
    for p in clips.rglob('*.mp4'):
        source=p.resolve()
        if source in active:continue
        assert source.is_relative_to(clips)
        dest=quarantine/source.relative_to(clips);assert dest.resolve().is_relative_to(quarantine);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():dest=dest.with_name(dest.stem+'_'+str(source.stat().st_mtime_ns)+dest.suffix)
        meta=source.with_suffix('.json');source.replace(dest)
        if meta.exists():meta.replace(dest.with_suffix('.json'))
        moves.append(dict(source=str(source),destination=str(dest)))
    journal=ROOT/'unpublished_clip_moves.json';prior=json.loads(journal.read_text()) if journal.exists() else []
    journal.write_text(json.dumps(prior+moves,indent=2));return ready,bad

def package(ready,bad,records):
    import platform,av,numpy as np,PIL
    environment=dict(python=platform.python_version(),av=av.__version__,numpy=np.__version__,Pillow=PIL.__version__,platform=platform.platform())
    (ROOT/'environment_preprocessing.json').write_text(json.dumps(environment,indent=2))
    (ROOT/'requirements-preprocessing.txt').write_text(f'av=={av.__version__}\nnumpy=={np.__version__}\nPillow=={PIL.__version__}\n')
    (ROOT/'processing_config.json').write_text(json.dumps(dict(reference_view='dashboard',mapping_sign='source_time = reference_time + offset',
        output_fps=30,output_size=[960,540],field_of_view='full source image',encoder='libx264',crf=20,preset='veryfast',
        max_chunk_frames=MAX_FRAMES,overlap_frames=OVERLAP,training_use=False,labels_created=False,
        script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')}),indent=2))
    m2dar=ROOT/'m2dar';m2dar.mkdir(exist_ok=True)
    with (m2dar/'A2_original_video_ids.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['video_id',*VIEWS])
        for r in records:w.writerow([r['video_id'],*[Path(r['views'][v]['path']).name for v in VIEWS]])
    (m2dar/'A2_original_videos.lst').write_text(''.join(Path(r['views'][v]['path']).as_posix()+'\n' for r in records for v in VIEWS))
    with (m2dar/'A2_segments_video_ids.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['video_id',*VIEWS])
        for g in ready:w.writerow([g['group_id'],*[Path(g['views'][v]['relative_output_path']).name for v in VIEWS]])
    (m2dar/'A2_segments_videos.lst').write_text(''.join(g['views'][v]['relative_output_path']+'\n' for g in ready for v in VIEWS))
    (ROOT/'original_recordings.json').write_text(json.dumps(records,indent=2))
    coverage=[]
    for r in records:
        spans=union([(g['reference_start_s'],g['reference_end_s']) for g in ready if g['video_id']==r['video_id']])
        duration=r['views']['dashboard']['duration_s'];holes=[];pos=0
        for a,b in spans:
            if a>pos:holes.append([pos,a])
            pos=max(pos,b)
        if pos<duration:holes.append([pos,duration])
        coverage.append(dict(video_id=r['video_id'],driver=r['driver'],session=r['session'],original_dashboard_duration_s=duration,
            included_reference_intervals_s=spans,excluded_or_unsupported_intervals_s=holes,included_reference_seconds=sum(b-a for a,b in spans)))
    (ROOT/'inference_coverage.json').write_text(json.dumps(coverage,indent=2))
    summary=dict(status='inference_ready_subset_pending_loader_verification',role='A2_unlabelled_evaluation_only',
        source_videos=90,source_recordings=30,drivers=15,prepared_triplets=len(ready),prepared_view_clips=3*len(ready),quarantined_triplets=len(bad),
        represented_recordings=len({g['video_id'] for g in ready}),unique_aligned_reference_seconds=sum(r['included_reference_seconds'] for r in coverage),
        original_reference_seconds=sum(r['original_dashboard_duration_s'] for r in coverage),
        prepared_GB=sum((ROOT/g['views'][v]['relative_output_path']).stat().st_size for g in ready for v in VIEWS)/1e9,
        fps=30,width=960,height=540,chunk_max_seconds=30,chunk_overlap_frames=64,labels_available=False,training_use=False,
        synchronization='Estimated visual-motion mapping; uncertain intervals and strong residual motion mismatches excluded.',
        metric_limitation='No local A2 action ground truth; cannot compute classification accuracy, mAP or official overlap score.')
    (ROOT/'READY.json').write_text(json.dumps(summary,indent=2))
    shutil.copy2(Path(r'F:\M2DAR_preprocessing\a2_dataset.py'),ROOT/'a2_dataset.py')
    shutil.copy2(Path(r'F:\M2DAR_preprocessing\restore_a2_times.py'),ROOT/'restore_a2_times.py')
    from create_previews import sheet,movie
    preview=ROOT/'preview';preview.mkdir(exist_ok=True);examples=[]
    for driver in sorted({g['driver'] for g in ready}):
        candidates=[g for g in ready if g['driver']==driver];g=candidates[len(candidates)//2]
        sheet(g,preview/f'driver_{driver}.jpg');examples.append(g['group_id'])
    if ready:movie(ready[0],preview/'synchronized_example.mp4')
    (preview/'examples.json').write_text(json.dumps(examples,indent=2))
    readme=f'''# A2 prepared for inference

The 90 original videos from 15 held-out drivers contain no supplied action labels. They are excluded from every training CSV and from the labelled A1 validation CSV. Do not create pseudo-labels or use A2 for training; this follows the 2024 Track 3 guidance. Local classification or localization accuracy cannot be computed without ground truth.

## Prepared media

{len(ready)} accepted aligned three-camera chunks ({3*len(ready)} MP4s, {summary['prepared_GB']:.3f} GB) have exact 30 fps timestamps at 960x540, retaining the full camera field of view. Chunks last at most 30 seconds and overlap by 64 frames so model proposals can cross chunk boundaries. They are inference windows, not independently labelled actions. {len(bad)} planned chunks were quarantined by post-export motion checks. See `inference_coverage.json` for retained and omitted time intervals, including any unavailable tails. Overlaps must not be counted twice when evaluating predictions.

Synchronization is estimated from visual motion; it is not a hardware-clock calibration. Short or static intervals can give inconclusive independent motion checks. Original burned-in clock digits can differ between cameras. Every retained MP4 has been fully decoded and checked for exact output frame count and timestamp spacing.

## Two inference paths

1. To reproduce the original M2DAR full-recording inference path, use `m2dar/A2_original_videos.lst` (90 absolute source paths) and `m2dar/A2_original_video_ids.csv` (official 2024 recording IDs with clean filenames). Original videos remain at their native 1920x1080/30 fps; no redundant full-video re-encode is required. Baseline inference performs its own spatial resize and normalization. This path preserves the original residual cross-camera timing mismatch.
2. To run on the prepared aligned media, use `m2dar/A2_segments_videos.lst` with `--video_dir F:/A2_preprocessed`. For the baseline Election stage, use `m2dar/A2_segments_video_ids.csv`; prediction pickle names are the listed MP4 basenames. Use the timing-restoration utility below before interpreting results on the original recording timeline. This is a modified inference protocol, so report it as such in the research comparison.

All mappings use dashboard seconds as the reference: `source_time = reference_time + offset`. `inference_groups.jsonl` records the original video ID, source times and offsets for each chunk. Do not concatenate chunks and discard these mappings. Unsupported gaps remain unknown; they are not labelled normal driving.

## Model input loader

```python
from a2_dataset import A2AlignedWindowDataset
dataset = A2AlignedWindowDataset(r"F:\\A2_preprocessed")
sample = dataset[0]
print(sample['video'].shape)  # (3, 3, 16, 448, 448)
print(sample['video_id'], sample['reference_start_s'])
```

The optional loader samples by exact prepared PTS, produces normalized RGB float32, and has no class label field. The full image is resized to 448 square, matching the baseline inference wrapper's `keep_scale=False` behavior. Set `as_torch=True` in your model environment if needed. No model inference or neural-network training was run during preparation.

## Restore segment prediction times

```text
python restore_a2_times.py --input segment_predictions.txt --output restored_candidates.jsonl
```

The input is M2DAR Election format: `segment_id class_id start_seconds end_seconds`. The JSONL output restores the official original video ID and dashboard seconds, clips to observed chunk bounds, and merges overlapping same-class candidates only within retained coverage. It does not invent predictions across unknown gaps. It preserves source-segment provenance. This utility prepares candidates; it does not compute a ground-truth score or tune the baseline's 2023 thresholds.

Original ID map: `F:/2024-data_video_ids.csv`. Relevant sources: [2024 FAQ](https://www.aicitychallenge.org/2024-faqs/), [M2DAR inference](https://github.com/PurdueDigitalTwin/M2DAR/blob/8fc6a3f76fc47207c4fa072dff7259f0debb5472/tools/inference/driver_action_rec.py), [M2DAR Election](https://github.com/PurdueDigitalTwin/M2DAR/blob/8fc6a3f76fc47207c4fa072dff7259f0debb5472/tools/inference/election.py).
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

def run():
    sources=json.loads((ROOT/'source_videos.json').read_text());ids,records=load_ids(sources)
    jobs=plan(sources,ids);print('A2 planned chunks',len(jobs),flush=True)
    if not jobs:raise RuntimeError('No supported A2 intervals')
    ready,bad=export(jobs)
    if not ready:raise RuntimeError('No A2 exports passed validation')
    package(ready,bad,records)

if __name__=='__main__':run()
