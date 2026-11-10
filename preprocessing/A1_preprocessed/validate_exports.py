"""Independently decode prepared clips and publish only validated training groups."""
from pathlib import Path
import sys,json,argparse,concurrent.futures,collections,time,csv
OUT=Path(__file__).resolve().parent;sys.path.insert(0,r'F:\A1_analysis\tools')
import av,numpy as np
from fit_alignment import correlations
VIEWS=['dashboard','rearview','right_side_window']

def decode_check(path,expected):
    times=[];motion=[];prev=None;corrupt=0;first=None;middle=None
    with av.open(str(path)) as c:
        if len(c.streams.video)!=1 or len(c.streams.audio)!=0:raise ValueError('Unexpected streams')
        s=c.streams.video[0];s.codec_context.thread_count=1
        if (s.width,s.height)!=(960,540):raise ValueError('Unexpected dimensions')
        for frame in c.decode(s):
            times.append(float(frame.time));corrupt+=int(frame.is_corrupt)
            a=frame.reformat(width=128,height=72,format='gray').to_ndarray().astype(np.float32)
            if prev is not None:
                d=np.maximum(np.abs(a-prev)-0.5,0)[:60]
                grid=d.reshape(4,15,8,16).mean(axis=(1,3)).ravel()
                motion.append([float(d.mean()),float(np.sort(grid)[-5:].mean())])
            prev=a
        duration=float(c.duration)/1e6
    if len(times)!=expected:raise ValueError(f'Frame count {len(times)} != {expected}')
    tt=np.asarray(times)
    if abs(tt[0])>1e-6 or np.max(np.abs(tt-np.arange(len(tt))/30))>1e-5:raise ValueError('Output PTS grid mismatch')
    if corrupt:raise ValueError(f'Corrupt decoded frames: {corrupt}')
    if abs(duration-expected/30)>0.002:raise ValueError('Duration mismatch')
    return np.asarray(motion),dict(frames=len(times),fps=30,duration_s=duration,corrupt_frames=corrupt,pts_checked=True)

def post_alignment(a,b):
    if len(a)<90:return dict(status='too_short_for_independent_motion_check',failed=False)
    margin=min(12,(len(a)-16)//2)
    if margin<3:return dict(status='too_short_for_independent_motion_check')
    score=[]
    for j in range(2):
        x=np.convolve(a[:,j],np.ones(3)/3,'same')[margin:-margin]
        y=np.convolve(b[:,j],np.ones(3)/3,'same')
        score.append(correlations(x,y))
    average=(score[0]+score[1])/2;peak=int(np.argmax(average));lag=peak-margin
    away=np.abs(np.arange(len(average))-peak)>=min(6,margin)
    prominence=float(average[peak]-average[away].max()) if away.any() else 0
    reliable=bool(average[peak]>=0.65 and prominence>=0.06)
    return dict(status='informative' if reliable else 'motion_check_inconclusive',residual_frames=lag if reliable else None,residual_s=lag/30 if reliable else None,correlation=float(average[peak]),peak_margin=prominence,pass_threshold_frames=3,failed=bool(reliable and abs(lag)>3))

def check_group(group):
    motions={};checks={}
    for v in VIEWS:
        p=OUT/group['views'][v]['relative_output_path'];m,checks[v]=decode_check(p,group['frame_count']);motions[v]=m
    alignment={v:post_alignment(motions['dashboard'],motions[v]) for v in VIEWS[1:]}
    failed=[v for v,r in alignment.items() if r.get('failed')]
    stats={v:dict(size_bytes=(OUT/group['views'][v]['relative_output_path']).stat().st_size,mtime_ns=(OUT/group['views'][v]['relative_output_path']).stat().st_mtime_ns) for v in VIEWS}
    return dict(group_id=group['group_id'],driver=group['driver'],session=group['session'],label=group['label'],split=group['split'],output_file_stats=stats,
        status='quarantined_post_export_motion' if failed else 'ready',video_checks=checks,alignment_checks=alignment,
        reason='Strong residual motion lag exceeds three frames' if failed else None)

def publish(plan,checks):
    extraction=json.loads((OUT/'dense_motion_progress.json').read_text())
    if extraction['done']!=60 or any(r.get('error') for r in extraction['results']):raise RuntimeError('The 60-video source pass must finish successfully before publication')
    mappings=json.loads((OUT/'alignment_estimates.json').read_text())['groups']
    if len(mappings)!=20:raise RuntimeError('All 20 source recordings must be accounted for')
    by={r['group_id']:r for r in checks};ready=[];quarantine=[]
    for group in plan:
        check=by.get(group['group_id'])
        if not check or check.get('error'):raise RuntimeError('Unvalidated export: '+group['group_id'])
        if check['status']=='ready':ready.append(dict(group,validation=check))
        else:quarantine.append(dict(group_id=group['group_id'],validation=check))
    train=[g for g in ready if g['split']=='train'];val=[g for g in ready if g['split']=='val']
    if not train or not val:raise RuntimeError('Both training and validation drivers are required')
    assert not ({g['driver'] for g in train}&{g['driver'] for g in val})
    if set(g['label'] for g in train)!=set(range(16)):raise RuntimeError('Training subset lacks one or more classes')
    m2dar=OUT/'m2dar';m2dar.mkdir(exist_ok=True)
    for name,groups in [('train',train),('val',val)]:
        with (OUT/f'{name}_groups.jsonl').open('w',encoding='utf-8') as f:
            for g in groups:f.write(json.dumps(g)+'\n')
        with (m2dar/f'{name}.csv').open('w',encoding='utf-8',newline='') as f:
            for g in groups:
                for v in VIEWS:f.write(f'{g["views"][v]["relative_output_path"]} {g["label"]}\n')
    with (OUT/'training_manifest.jsonl').open('w',encoding='utf-8') as f:
        for g in ready:f.write(json.dumps(g)+'\n')
    (OUT/'quarantined_exports.json').write_text(json.dumps(quarantine,indent=2))
    summary=dict(status='training_ready_subset',source_recordings=20,prepared_triplets=len(ready),prepared_view_clips=3*len(ready),quarantined_triplets=len(quarantine),
        train_drivers=sorted({g['driver'] for g in train}),val_drivers=sorted({g['driver'] for g in val}),
        train_triplets=len(train),val_triplets=len(val),train_classes=dict(collections.Counter(g['label'] for g in train)),val_classes=dict(collections.Counter(g['label'] for g in val)),
        source_reference_hours=sum(g['frame_count']/30 for g in ready)/3600,
        output_width=960,output_height=540,output_fps=30,model_frames=16,model_sampling_stride=4,
        verification='Every included clip fully decoded, equal frame counts and exact output PTS checked; strong post-export motion residuals above three frames quarantined.',
        limitation='Alignment is estimated from visual motion and reviewed examples, not hardware-clock ground truth. Uninformative motion cannot establish frame-level accuracy. Uncertain source intervals were excluded.')
    (OUT/'READY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit-groups',type=int);p.add_argument('--publish',action='store_true');p.add_argument('--existing-only',action='store_true');args=p.parse_args()
    plan=json.loads((OUT/'clip_plan.json').read_text())['groups'];jobs=plan[:args.limit_groups] if args.limit_groups else plan
    if args.existing_only:jobs=[g for g in jobs if all((OUT/g['views'][v]['relative_output_path']).exists() for v in VIEWS)]
    cache_path=OUT/'validation_cache.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    results=[];pending_jobs=[];start=time.time()
    for g in jobs:
        old=cache.get(g['group_id']);unchanged=bool(old and not old.get('error'))
        for v in VIEWS:
            pth=OUT/g['views'][v]['relative_output_path']
            if not pth.exists():unchanged=False;break
            st=pth.stat();previous=old.get('output_file_stats',{}).get(v,{}) if old else {}
            if previous.get('size_bytes')!=st.st_size or previous.get('mtime_ns')!=st.st_mtime_ns:unchanged=False
        if unchanged:results.append(old)
        else:pending_jobs.append(g)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(check_group,g):g for g in pending_jobs}
        for future in concurrent.futures.as_completed(futures):
            try:r=future.result()
            except Exception as e:r=dict(group_id=futures[future]['group_id'],error=repr(e),status='error')
            results.append(r)
            cache[r['group_id']]=r;cache_path.write_text(json.dumps(cache,indent=2))
            if len(results)%20==0 or len(results)==len(jobs):print('CHECKED',len(results),'/',len(jobs),'elapsed',round(time.time()-start,1),flush=True)
    name='pilot_validation.json' if args.limit_groups else 'export_validation.json'
    (OUT/name).write_text(json.dumps(results,indent=2))
    print('Validation statuses:',dict(collections.Counter(r['status'] for r in results)),flush=True)
    if args.publish:publish(plan,results)
