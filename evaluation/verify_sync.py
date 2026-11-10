"""Model-blind synchronization assessment; preserves the primary test index."""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,importlib.util,json,math,multiprocessing,shutil,sys,time,traceback,zipfile
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sync';BASE=Path('F:/A1_preprocessed');FPS=30
sys.path.insert(0,'F:/A1_analysis/tools');sys.path.insert(0,str(BASE))
import av,numpy as np
from evaluate_test import verify_test,stable,digest
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');t.replace(p)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def initialize_worker(scratch_queue=None):
 global EXTRACT,VALIDATE,MEDIA,WORKER_SCRATCH
 MEDIA=Path(read(ROOT/'READY.json')['media_root'])
 EXTRACT=module('test_dense_extract','F:/A1_2_preprocessed/extract_motion.py');EXTRACT.OUT=OUT
 VALIDATE=module('test_alignment_validate',BASE/'validate_exports.py')
 WORKER_SCRATCH=Path(scratch_queue.get()) if scratch_queue is not None else OUT/'.staging'
 WORKER_SCRATCH.mkdir(parents=True,exist_ok=True)
def extract_source(v):
 target=OUT/'dense_motion'/f'{v["key"]}.npz';completed=read(ROOT/'completed'/f'{v["key"]}.json');expected=completed['source_sha256']
 if target.exists():
  try:
   with np.load(target,allow_pickle=False) as z:meta=json.loads(str(z['metadata']));z['t'];z['f']
   if meta.get('source_sha256')==expected and meta.get('complete'):return dict(key=v['key'],cached=True,metadata=meta)
  except Exception:
   target.replace(target.with_suffix('.interrupted'))
 raw=WORKER_SCRATCH/f'{v["key"]}.mp4'
 try:
  if raw.exists() and digest(raw)!=expected:raw.unlink()
  if not raw.exists():
   if shutil.disk_usage(raw.parent).free<v['source_size_bytes']+600*1024**2:raise RuntimeError('Scratch disk reserve reached')
   with zipfile.ZipFile(v['archive_path']) as z,z.open(v['archive_member']) as src,raw.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
  if digest(raw)!=expected:raise ValueError('Source bytes changed')
  r=dict(driver=v['driver'],session=v['session'],view=v['view'],path=str(raw),duration_s=completed['source_duration_s'])
  result=EXTRACT.process(r,None,None)
  with np.load(target,allow_pickle=False) as z:t=z['t'].copy();f=z['f'].copy();meta=json.loads(str(z['metadata']))
  meta['source_sha256']=expected
  if not meta['complete'] or meta.get('corrupt_frames',0):raise ValueError('Full-source motion decoding did not pass')
  np.savez_compressed(target,t=t,f=f,metadata=json.dumps(meta))
  return dict(key=v['key'],cached=False,metadata=meta)
 finally:
  if raw.exists():
   assert raw.resolve().parent==WORKER_SCRATCH.resolve();raw.unlink()
def prepared_motion(row):
 target=OUT/'clip_motion'/f'{row["sample_id"]}.npz'
 if target.exists():
  with np.load(target,allow_pickle=False) as z:
   if str(z['sha256'])==row['sha256']:return row['sample_id']
 path=MEDIA/row['path']
 if digest(path)!=row['sha256']:raise ValueError('Prepared test clip changed')
 motion,checks=VALIDATE.decode_check(path,row['frames'])
 np.savez_compressed(target,m=motion,sha256=row['sha256'],metadata=json.dumps(checks));return row['sample_id']
def plan_groups(rows):
 per=collections.defaultdict(dict)
 for r in rows:
  key=(r['driver'],r['session'],r['label'])
  if r['view'] in per[key]:raise ValueError('Ambiguous action correspondence; no guessed pairing')
  per[key][r['view']]=r
 mappings=read(OUT/'alignment_estimates.json')['groups'];plan={};omissions=[]
 for g in mappings:
  d,s=g['driver'],g['session'];vv=g['views']
  for label in range(16):
   rr=per.get((d,s,label),{})
   if set(rr)!={'dashboard','rearview','right_side_window'}:continue
   found=False
   for rear in vv['rearview']['segments']:
    for right in vv['right_side_window']['segments']:
     offsets=dict(dashboard=0,rearview=rear['offset_s'],right_side_window=right['offset_s'])
     lo=max(rr['dashboard']['original_label_start_s'],rear['reference_start_s'],right['reference_start_s'],*[rr[v]['original_label_start_s']-offsets[v] for v in rr])+.25
     hi=min(rr['dashboard']['original_label_end_s'],rear['reference_end_s'],right['reference_end_s'],*[rr[v]['original_label_end_s']-offsets[v] for v in rr])-.25
     begin=math.ceil(lo*30-1e-7);end=math.floor(hi*30+1e-7);n=end-begin
     if n<16:continue
     specs={};valid=True
     for view,row in rr.items():
      start=int(round(begin+offsets[view]*30-row['source_start_s']*30))
      if start<0 or start+n>row['frames']:valid=False;break
      specs[view]=dict(sample_id=row['sample_id'],path=row['path'],sha256=row['sha256'],start_frame=start,end_frame=start+n,source_offset_frames=round(offsets[view]*30))
     if not valid:continue
     identity=dict(driver=d,session=s,label=label,reference_start_frame=begin,reference_end_frame=end,views=specs)
     gid='sync_'+stable(identity)[:20]
     plan[gid]=dict(group_id=gid,**identity,frame_count=n,fps=30,split='test',source_alignment_anchors={v:vv[v]['segments'] for v in ['rearview','right_side_window']});found=True
   if not found:omissions.append(dict(driver=d,session=s,label=label,reason='No common labeled interval with supported source alignment and minimum clip duration'))
 return sorted(plan.values(),key=lambda x:(x['driver'],x['session'],x['reference_start_frame'])),omissions
def run(workers):
 global EXTRACT,VALIDATE,MEDIA
 MEDIA=Path(read(ROOT/'READY.json')['media_root']);protocol,rows=verify_test(MEDIA,check_media=False)
 OUT.mkdir(exist_ok=True)
 for name in ['dense_motion','clip_motion','.staging']:(OUT/name).mkdir(exist_ok=True)
 policy=dict(schema_version=1,primary_test_protocol_sha256=stable(protocol),primary_test_index_sha256=digest(MEDIA/'test_index.jsonl'),model_predictions_used=False,source_extractor_sha256=digest(Path('F:/A1_2_preprocessed/extract_motion.py')),source_fit_sha256=digest(BASE/'fit_alignment.py'),output_checker_sha256=digest(BASE/'validate_exports.py'),source_fit='Existing fixed full-source PTS/global+top-grid motion policy; +/-45-frame search, conservative piecewise constants and unsupported-interval exclusion',group_policy='Unique corresponding camera/driver/session/class only, supplied-label common overlap, 0.25s edges, at least16 frames',fusion_eligibility='Both post-crop camera checks must be informative with residual <=3 frames. Inconclusive and failed groups are excluded.',limitation='Estimated visual synchronization, not hardware-clock ground truth. The primary per-camera test index is unchanged.')
 if (OUT/'sync_protocol.json').exists():
  if read(OUT/'sync_protocol.json')!=policy:raise ValueError('Frozen synchronization policy changed')
 else:write(OUT/'sync_protocol.json',policy)
 EXTRACT=module('test_dense_extract','F:/A1_2_preprocessed/extract_motion.py');EXTRACT.OUT=OUT
 VALIDATE=module('test_alignment_validate',BASE/'validate_exports.py')
 sources=read(ROOT/'source_plan.json');source_rows=[]
 for v in sources:
  completed=read(ROOT/'completed'/f'{v["key"]}.json')
  source_rows.append(dict(driver=v['driver'],session=v['session'],view=v['view'],path=str(OUT/'.staging'/f'{v["key"]}.mp4'),duration_s=completed['source_duration_s']))
 write(OUT/'source_videos.json',source_rows)
 completed=[];began=time.time()
 context=multiprocessing.get_context('spawn');scratch_queue=context.Queue()
 secondary_scratch=Path('E:/IVP_GROUP16/M2DAR_sync_scratch');secondary_scratch.mkdir(exist_ok=True)
 for i in range(workers):scratch_queue.put(str(OUT/'.staging' if i%2==0 else secondary_scratch))
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize_worker,initargs=(scratch_queue,)) as pool:
  futures={pool.submit(extract_source,v):v for v in sources}
  for future in concurrent.futures.as_completed(futures):
   result=future.result();completed.append(result)
   write(OUT/'progress.json',dict(stage='full_source_motion',completed=len(completed),total=len(sources),elapsed_s=time.time()-began,last_source=result['key']))
   print('SOURCE',len(completed),'/',len(sources),result['key'],round(time.time()-began,1),flush=True)
 write(OUT/'full_source_motion_audit.json',completed)
 fit=module('test_source_alignment',BASE/'fit_alignment.py');fit.OUT=OUT;fit.run()
 plan,omissions=plan_groups(rows);write(OUT/'candidate_groups.json',plan);write(OUT/'alignment_omissions.json',omissions)
 needed={s['sample_id'] for g in plan for s in g['views'].values()};selected=[r for r in rows if r['sample_id'] in needed]
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=initialize_worker) as pool:
  futures=[pool.submit(prepared_motion,r) for r in selected]
  for i,future in enumerate(concurrent.futures.as_completed(futures),1):
   future.result()
   if i%25==0 or i==len(selected):
    write(OUT/'progress.json',dict(stage='independent_prepared_clip_check',completed=i,total=len(selected)));print('CLIP CHECK',i,'/',len(selected),flush=True)
 accepted=[];checks=[]
 for g in plan:
  motion={}
  for v,s in g['views'].items():
   with np.load(OUT/'clip_motion'/f'{s["sample_id"]}.npz',allow_pickle=False) as z:motion[v]=z['m'][s['start_frame']:s['end_frame']-1].copy()
   if len(motion[v])!=g['frame_count']-1:raise ValueError('Cropped motion length mismatch')
  pair={v:VALIDATE.post_alignment(motion['dashboard'],motion[v]) for v in ['rearview','right_side_window']}
  passed=all(c['status']=='informative' and not c.get('failed',False) and abs(c['residual_frames'])<=3 for c in pair.values())
  check=dict(group_id=g['group_id'],driver=g['driver'],session=g['session'],label=g['label'],verified=passed,post_crop_alignment=pair);checks.append(check)
  if passed:accepted.append(dict(g,verification=check))
 write(OUT/'group_checks.json',checks)
 text=''.join(json.dumps(x)+'\n' for x in accepted);(OUT/'synchronized_test_groups.jsonl').write_text(text,encoding='utf-8')
 report=dict(status='synchronization_assessment_complete',primary_test_unchanged=True,source_videos_checked=len(sources),source_recordings=len({(v['driver'],v['session']) for v in sources}),candidate_aligned_groups=len(plan),verified_groups=len(accepted),excluded_or_inconclusive_groups=len(plan)-len(accepted),verified_drivers=sorted({g['driver'] for g in accepted}),verified_classes=sorted({g['label'] for g in accepted}),class_counts=dict(collections.Counter(g['label'] for g in accepted)),protocol_sha256=stable(policy),primary_index_sha256=policy['primary_test_index_sha256'],groups_sha256=digest(OUT/'synchronized_test_groups.jsonl'),source_motion_audit_sha256=digest(OUT/'full_source_motion_audit.json'),group_checks_sha256=digest(OUT/'group_checks.json'),alignment_sha256=digest(OUT/'alignment_estimates.json'),model_predictions_used=False,hardware_clock_ground_truth=False)
 write(OUT/'READY_SYNC.json',report);write(OUT/'progress.json',report);print('SYNC COMPLETE',json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 try:run(a.workers)
 except Exception as e:write(OUT/'progress.json',dict(status='stopped',error=str(e)));traceback.print_exc();raise SystemExit(1)
