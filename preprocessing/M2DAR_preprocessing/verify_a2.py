"""Verify A2 media provenance, inference-only isolation and model input loading."""
from pathlib import Path
import sys,json,hashlib,collections
ROOT=Path(r'F:\A2_preprocessed');sys.path.insert(0,str(ROOT));sys.path.insert(0,r'F:\A1_analysis\tools')
import numpy as np
from a2_dataset import A2AlignedWindowDataset
from restore_a2_times import restore

def run():
    sources=json.loads((ROOT/'source_inventory_original.json').read_text());assert len(sources)==90
    assert len({r['driver'] for r in sources})==15
    groups=[json.loads(x) for x in (ROOT/'inference_groups.jsonl').read_text().splitlines() if x.strip()]
    assert groups and all(g['label'] is None and g['split']=='inference' and not g['action_ground_truth_available'] for g in groups)
    for r in sources:
        p=Path(r['path']);st=p.stat();assert st.st_size==r['source_size_bytes'] and st.st_mtime_ns==r['source_mtime_ns'],p
        h=hashlib.sha256()
        with p.open('rb') as f:
            for pos in [0,max(0,st.st_size//2-32768),max(0,st.st_size-65536)]:f.seek(pos);h.update(f.read(65536))
        assert h.hexdigest()==r['sampled_source_fingerprint_sha256'],p
        with np.load(ROOT/'dense_motion'/f"{r['driver']}_{r['session']}_{r['view']}.npz",allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));assert meta['complete'] and not meta['corrupt_frames']
    active=set()
    for g in groups:
        assert g['validation']['status']=='ready'
        for v in g['views']:
            path=ROOT/g['views'][v]['relative_output_path'];assert path.is_file();assert str(path.resolve()) not in active;active.add(str(path.resolve()))
            check=g['validation']['video_checks'][v];assert check['frames']==g['frame_count'] and check['pts_checked'] and check['corrupt_frames']==0
    assert len(list((ROOT/'clips').rglob('*.mp4')))==3*len(groups)
    assert not (ROOT/'m2dar'/'train.csv').exists() and not (ROOT/'m2dar'/'val.csv').exists()
    assert len((ROOT/'m2dar'/'A2_original_videos.lst').read_text().splitlines())==90
    assert len((ROOT/'m2dar'/'A2_segments_videos.lst').read_text().splitlines())==3*len(groups)
    coverage=json.loads((ROOT/'inference_coverage.json').read_text())
    originals={r['video_id']:r for r in json.loads((ROOT/'original_recordings.json').read_text())}
    fallback=[]
    for r in coverage:
        missing=sum(b-a for a,b in r['excluded_or_unsupported_intervals_s'])
        assert abs(r['included_reference_seconds']+missing-r['original_dashboard_duration_s'])<1e-5
        for a,b in r['excluded_or_unsupported_intervals_s']:
            fallback.append(dict(video_id=r['video_id'],driver=r['driver'],session=r['session'],reference_start_s=a,reference_end_s=b,
                dashboard_source_path=originals[r['video_id']]['views']['dashboard']['path'],role='unlabelled_inference_only',
                synchronized_three_view_input_available=False,action_label=None))
    (ROOT/'unsupported_interval_inference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fallback))
    current_sources=json.loads((ROOT/'source_videos.json').read_text())
    source_health=dict(source_videos=90,decoded_corrupt_frames=0,irregular_pts_files=sum(r.get('irregular_pts_count',0)>0 for r in current_sources),
        nonzero_start_pts_files=sum(r.get('first_pts_s',0)!=0 for r in current_sources),
        native_resolution=[1920,1080],native_nominal_fps=30,
        full_video_reencode_needed_for_frame_rate=any(r.get('irregular_pts_count',0)>0 or r.get('first_pts_s',0)!=0 or abs(r['fps']-30)>1e-6 for r in current_sources))
    (ROOT/'source_health_summary.json').write_text(json.dumps(source_health,indent=2))
    ds=A2AlignedWindowDataset(ROOT);selected={0,len(ds)-1};buckets=collections.defaultdict(list)
    for i,(gi,start) in enumerate(ds.windows):buckets[ds.groups[gi]['video_id']].append(i)
    for rr in buckets.values():selected.update([rr[0],rr[len(rr)//2],rr[-1]])
    checks=[]
    for index in sorted(selected):
        item=ds[index];a=item['video'];assert a.shape==(3,3,16,448,448) and a.dtype==np.float32 and np.isfinite(a).all()
        assert 'label' not in item
        g=next(g for g in ds.groups if g['group_id']==item['group_id'])
        assert np.all(item['reference_sample_times_s']>=g['reference_start_s']-1e-6)
        assert np.all(item['reference_sample_times_s']<g['reference_end_s']+1e-6)
        checks.append(dict(index=index,group_id=item['group_id'],video_id=item['video_id'],shape=list(a.shape),passed=True))
    synthetic=[dict(group_id='a',video_id=1,frame_count=300,reference_start_s=100),dict(group_id='b',video_id=1,frame_count=300,reference_start_s=108),dict(group_id='c',video_id=1,frame_count=300,reference_start_s=130)]
    restored=restore(['a 3 7 12','b 3 0 3','c 3 0 2','a 0 0 0'],synthetic)
    assert len(restored)==2 and restored[0]['start_s']==107 and restored[0]['end_s']==111 and restored[1]['start_s']==130,restored
    result=dict(source_files_checked=90,source_size_mtime_and_sampled_fingerprints_unchanged=True,all_active_media_manifested=True,
        action_labels_available=False,training_use=False,model_training_executed=False,model_inference_executed=False,
        model_input_windows_checked=len(checks),represented_recordings=len(buckets),model_input_checks=checks,
        timestamp_restoration_boundary_overlap_and_gap_checks_passed=True,coverage_partitions_reconcile_to_full_original_duration=True,
        unsupported_interval_references=len(fallback))
    (ROOT/'inference_smoke_test.json').write_text(json.dumps(result,indent=2))
    ready=json.loads((ROOT/'READY.json').read_text());ready.update(status='inference_ready_subset',verified_model_input_windows=len(checks),source_provenance_checks_passed=True,
        aligned_reference_time_coverage_fraction=sum(r['included_reference_seconds'] for r in coverage)/sum(r['original_dashboard_duration_s'] for r in coverage),
        unsupported_interval_references=len(fallback))
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8')
    marker='## Full-recording evaluation coverage'
    if marker not in text:
        p.write_text(text+'''\n## Full-recording evaluation coverage

The aligned chunks are a conservative subset. Do not calculate or present a full-A2 benchmark score by evaluating only retained intervals. The original full-recording inference list preserves all 90 inputs. `unsupported_interval_inference.jsonl` points to original dashboard footage for every omitted or uncertain interval, without asserting cross-camera synchronization or assigning action labels. A full-coverage model must use an explicitly documented fallback (for example, dashboard-only inference) in those intervals, or otherwise account for them; they must not silently become normal-driving predictions. The preprocessing records these references but does not run a fallback neural model. `inference_coverage.json` reconciles retained and unsupported times to each full original dashboard duration.
''',encoding='utf-8')
    (ROOT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='model_input_checks'},indent=2))

if __name__=='__main__':run()
