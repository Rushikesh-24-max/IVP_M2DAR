"""Freeze and prepare a driver-disjoint A1 test set from read-only ZIP archives.

Only supplied per-camera labels are used. No model predictions are read.
"""
from __future__ import annotations
import argparse,collections,csv,hashlib,importlib.util,json,math,os,re,shutil,sys,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'F:/A1_analysis/tools')
import av
LABELS=Path('E:/IVP_GROUP16/2024-data_labelsAndinstructions/labels&instructions/A1/A1')
SPLITS=Path('F:/M2DAR_preprocessed/splits.json')
ARCHIVES=[Path('E:/IVP_GROUP16')/f'A1_{i}.zip' for i in [5,6]]
CHECKPOINT='f2a254dca49cd5156b344d3abff40e21061bb3fc4e605ba1728925d2d3db3115'
VIEWS=('dashboard','rearview','right_side_window')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def identify(name):
 s=Path(name.replace('\\','/')).name.lower().replace('noaudio_','')
 m=re.search(r'user_id_(\d+)_(\d+)',s)
 if not m:raise ValueError('Unrecognized recording filename: '+name)
 view='dashboard' if s.startswith('dashboard') else 'rearview' if s.startswith('rear') else 'right_side_window' if s.startswith('right') else None
 if not view:raise ValueError('Unrecognized camera')
 return m[1],m[2],view
def seconds(value):
 p=[int(x) for x in value.strip().split(':')]
 if len(p)!=3 or p[0]<0 or not 0<=p[1]<60 or not 0<=p[2]<60:raise ValueError('Malformed timestamp')
 return p[0]*3600+p[1]*60+p[2]
def source_timing_evidence(video,fps,source_hash):
 if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid source average FPS')
 if abs(fps-30)<=.01:return None
 # Frozen fps=30 is the OUTPUT grid. A metadata average is not a reliable
 # source-time clock when extra frames share near-identical PTS. Require a
 # full source decode audit, tied to its exact bytes, before allowing this.
 audit_path=ROOT/'fps_investigation.json'
 audits=read(audit_path) if audit_path.exists() else []
 evidence=next((x for x in audits if x['key']==video['key'] and x['source_sha256']==source_hash),None)
 if not evidence:raise ValueError('Nonstandard average source FPS requires a full decoded timing audit')
 if evidence['corrupt_frames'] or evidence['nonincreasing_pts'] or abs(evidence['median_frame_interval_s']-1/30)>.002 or evidence['max_frame_interval_s']>1/30+.002:
  raise ValueError('Source timing audit failed; investigate separately')
 return dict(evidence,decision='Accepted using decoded PTS; frozen 30-FPS output and per-sample timing-error bound unchanged')
def annotations():
 rows=[];hashes={};container_issues=[]
 for p in sorted(LABELS.glob('user_id_*.csv')):
  hashes[str(p)]=sha(p);previous=None;ids=set()
  with p.open(encoding='utf-8-sig',newline='') as f:
   for number,r in enumerate(csv.DictReader(f),2):
    name=(r.get('Filename') or previous or '').strip();previous=name
    try:
     d,s,v=identify(name);ids.add(d)
     label=re.fullmatch(r'Class\s*(\d+)',r['Label (Primary)'].strip(),re.I)
     if not label:raise ValueError('Invalid class')
     c=int(label[1]);a,b=seconds(r['Start Time']),seconds(r['End Time'])
     declared=re.sub('[^a-z]','',r['Camera View'].lower());dv='dashboard' if declared.startswith('dash') else 'rearview' if declared.startswith('rear') else 'right_side_window' if declared.startswith('right') else None
     problems=[]
     if not 0<=c<16:problems.append('invalid_class')
     if a>=b:problems.append('nonpositive_interval')
     if dv!=v:problems.append('camera_column_filename_conflict')
     rows.append(dict(driver=d,session=s,view=v,label=c,start_s=a,end_s=b,source_annotation_file=str(p),source_annotation_sha256=hashes[str(p)],source_annotation_row=number,original_filename=name,annotation_container_driver=p.stem.removeprefix('user_id_'),issues=problems))
    except (ValueError,KeyError) as e:container_issues.append(dict(file=str(p),row=number,reason=str(e)))
  if ids!={p.stem.removeprefix('user_id_')}:container_issues.append(dict(file=str(p),container_driver=p.stem.removeprefix('user_id_'),recording_drivers=sorted(ids),reason='CSV container ID differs from explicitly named recordings; filename joins only, never inferred renaming'))
 return rows,hashes,container_issues
def freeze(media_root):
 existing=read(ROOT/'protocol_lock.json') if (ROOT/'protocol_lock.json').exists() else None
 splits=read(SPLITS);used=set(splits['train']+splits['val']);rows,hashes,container_issues=annotations();videos=[];seen=set()
 # Conservative cross-part duplicate screening; CRC+size is not a cryptographic identity proof.
 prior=set()
 for i in range(1,5):
  with zipfile.ZipFile(Path('E:/IVP_GROUP16')/f'A1_{i}.zip') as z:
   prior.update((x.CRC,x.file_size) for x in z.infolist() if x.filename.lower().endswith('.mp4') and not x.filename.startswith('__MACOSX/'))
 for archive in ARCHIVES:
  st=archive.stat()
  with zipfile.ZipFile(archive) as z:
   for info in z.infolist():
    if not info.filename.lower().endswith('.mp4') or info.filename.startswith('__MACOSX/'):continue
    d,s,v=identify(info.filename);key=(d,s,v)
    if key in seen:raise ValueError('Duplicate source recording key: '+str(key))
    seen.add(key)
    if d in used:raise ValueError('Driver leakage: '+d)
    if (info.CRC,info.file_size) in prior:raise ValueError('Potential duplicate of train/validation raw media; investigate before freezing')
    videos.append(dict(driver=d,session=s,view=v,archive_path=str(archive),archive_member=info.filename,archive_size_bytes=st.st_size,archive_mtime_ns=st.st_mtime_ns,source_size_bytes=info.file_size,source_crc32=info.CRC))
 actual={r['driver'] for r in videos};matched={d for d in actual if any(r['driver']==d for r in rows)}
 excluded=[dict(driver=d,reason='No CSV row explicitly names this raw recording driver; no guessed relabeling') for d in sorted(actual-matched)]
 videos=sorted([r for r in videos if r['driver'] in matched],key=lambda r:(r['driver'],r['session'],r['view']))
 keys={(v['driver'],v['session'],v['view']) for v in videos}
 chosen=[dict(r) for r in rows if (r['driver'],r['session'],r['view']) in keys]
 grouped=collections.defaultdict(list)
 for r in chosen:grouped[(r['driver'],r['session'],r['view'])].append(r)
 for key,rr in grouped.items():
  identities=set()
  for r in rr:
   identity=(r['label'],r['start_s'],r['end_s'])
   if identity in identities:r['issues'].append('duplicate_annotation')
   identities.add(identity)
   inside=[y for y in rr if y is not r and r['start_s']<y['start_s']<r['end_s'] and y['label']!=r['label']]
   if len(inside)>=2:r['issues'].append('spans_multiple_labeled_actions')
  valid=[r for r in rr if not r['issues']]
  for i,r in enumerate(valid):
   for q in valid[i+1:]:
    if max(r['start_s'],q['start_s'])<min(r['end_s'],q['end_s']):
     r['issues'].append('overlapping_annotations');q['issues'].append('overlapping_annotations')
 for video in videos:
  video['key']='_'.join([video['driver'],video['session'],video['view']]);video['annotations']=grouped[(video['driver'],video['session'],video['view'])]
 protocol=dict(schema_version=1,purpose='Independent A1 driver-held-out classification test for the fixed epoch-170 checkpoint',membership_rule='All raw drivers in A1_5/A1_6 with explicit matching annotation recording filenames and no overlap with existing training/validation drivers',test_drivers=sorted(matched),excluded_raw_drivers=excluded,training_drivers=sorted(splits['train']),validation_drivers=sorted(splits['val']),checkpoint_sha256=CHECKPOINT,selection_uses_model_predictions=False,label_source='Supplied A1 CSVs; no AI annotations or inferred normal gaps',preprocessing=dict(width=960,height=540,fps=30,encoder='libx264',crf=20,preset='veryfast',edge_margin_s=.25,minimum_frames=16,source_time_selection='nearest presentation timestamp; maximum error <= 1/30+0.002 seconds',short_clip_policy='Keep valid frames; existing model sampler may repeat boundary frame'),evaluation=dict(primary='Single-camera pooled macro-F1',secondary=['top1_accuracy','macro_precision','macro_recall_balanced_accuracy','top5_accuracy','per_class','per_camera','per_driver'],temporal_sampling='Same deterministic center-window/crop as existing validation',three_camera_fusion='Not authorized until synchronization and corresponding triplets have been verified',checkpoint_selection='Frozen best epoch170; never tune on these test outputs'),split_input_sha256=sha(SPLITS),annotation_sha256=hashes,exporter_sha256=sha(Path('F:/A1_preprocessed/export_clips.py')),source_plan_sha256=stable(videos),media_root=str(media_root.resolve()),source_videos=len(videos),source_recordings=len({(v['driver'],v['session']) for v in videos}),container_issues=container_issues,screening=dict(driver_overlap=[],crc_size_collisions_with_training_archives=0,limitation='Driver IDs and archive metadata used; CRC+size screening is not full source-byte hashing across all training videos. Test source SHA256 is computed during extraction.'))
 if existing:
  if existing!=protocol:raise ValueError('Frozen protocol or inputs changed. Preserve this test set and investigate; do not silently rebuild membership.')
 else:write(ROOT/'protocol_lock.json',protocol);write(ROOT/'source_plan.json',videos)
 write(ROOT/'label_container_issues.json',container_issues)
 write(ROOT/'selection_summary.json',dict(status='membership_frozen_not_yet_prepared',test_drivers=sorted(matched),raw_drivers_found=len(actual),excluded_raw_drivers=excluded,source_videos=len(videos),candidate_annotation_rows=len(chosen),annotation_issue_counts=dict(collections.Counter(i for r in chosen for i in set(r['issues']))),source_label_files_used=sorted({r['source_annotation_file'] for r in chosen}),driver_leakage=False,media_root=protocol['media_root']))
 return protocol,videos
def verify_clip(path,expected):
 count=0;last=None
 with av.open(str(path)) as c:
  stream=c.streams.video[0]
  if stream.width!=960 or stream.height!=540 or abs(float(stream.average_rate)-30)>.001:raise ValueError('Invalid output dimensions/FPS')
  for f in c.decode(video=0):
   if f.is_corrupt or f.time is None or abs(float(f.time)-count/30)>1e-4:raise ValueError('Corrupt frame or output PTS mismatch')
   count+=1;last=float(f.time)
 if count!=expected:raise ValueError(f'Decoded {count}, expected {expected}')
 return dict(decoded_frames=count,corrupt_frames=0,pts_verified=True,last_pts_s=last,sha256=sha(path),size_bytes=path.stat().st_size)
def prepare_video(video,protocol,media_root,scratch,exporter):
 key=video['key'];done=ROOT/'completed'/f'{key}.json';signature=stable(dict(video=video,protocol_sha256=stable(protocol)))
 if done.exists():
  prior=read(done)
  if prior['signature']!=signature:raise ValueError('Saved source signature mismatch')
  for r in prior['clips']:
   path=media_root/r['path']
   if not path.is_file() or sha(path)!=r['sha256']:raise ValueError('Completed test clip missing or changed')
  return prior
 if shutil.disk_usage(media_root).free<1024**3:raise RuntimeError('Less than 1 GiB free on the test-media drive; paused without deleting source data')
 if shutil.disk_usage(scratch).free<video['source_size_bytes']+700*1024**2:raise RuntimeError('Insufficient temporary workspace; original archives remain unchanged')
 raw=scratch/f'{key}.mp4';temp=raw.with_suffix('.extracting');source_hash=hashlib.sha256()
 try:
  archive=Path(video['archive_path']);st=archive.stat()
  if st.st_size!=video['archive_size_bytes'] or st.st_mtime_ns!=video['archive_mtime_ns']:raise ValueError('Input archive changed')
  print('EXTRACT',key,flush=True)
  with zipfile.ZipFile(archive) as z, z.open(video['archive_member']) as src,temp.open('wb') as out:
   for b in iter(lambda:src.read(4*1024**2),b''):source_hash.update(b);out.write(b)
  if temp.stat().st_size!=video['source_size_bytes']:raise ValueError('Extracted source size mismatch')
  temp.replace(raw)
  with av.open(str(raw)) as c:
   stream=c.streams.video[0];duration=float(stream.duration*stream.time_base) if stream.duration is not None else c.duration/1e6;fps=float(stream.average_rate)
   timing_audit=source_timing_evidence(video,fps,source_hash.hexdigest())
  clips=[];exclusions=[];annotations=video['annotations']
  for row in annotations:
   issues=list(set(row['issues']))
   if row['start_s']<0 or row['start_s']>=duration or row['end_s']>duration+1e-5:issues.append('outside_video_duration')
   start=math.ceil((row['start_s']+.25)*30-1e-7);end=math.floor((row['end_s']-.25)*30+1e-7);frames=end-start
   if frames<16:issues.append('too_short_after_fixed_edge_margin')
   if issues:exclusions.append(dict(row,issues=sorted(set(issues))));continue
   gid='test_'+stable(dict(video=key,row=row['source_annotation_row'],file=row['source_annotation_sha256'],start=start,end=end,label=row['label']))[:20]
   relative=f'clips/{video["driver"]}/{video["session"]}/{gid}_{video["view"]}.mp4'
   spec=dict(source_path=str(raw),source_start_s=start/30,source_end_s=end/30,offset_s=0,relative_output_path=relative)
   job=dict(group_id=gid,driver=video['driver'],session=video['session'],label=row['label'],split='test',fps=30,frame_count=frames,reference_start_s=start/30,views={video['view']:spec})
   if shutil.disk_usage(media_root).free<700*1024**2:raise RuntimeError('Test media disk reserve reached; completed clips are retained')
   exported=exporter.export_view(job,video['view']);path=media_root/relative
   verified=verify_clip(path,frames)
   clips.append(dict(sample_id=gid,group_id=gid,driver=video['driver'],session=video['session'],view=video['view'],label=row['label'],split='test',path=relative,frames=frames,fps=30,width=960,height=540,source_start_s=start/30,source_end_s=end/30,original_label_start_s=row['start_s'],original_label_end_s=row['end_s'],source_annotation_file=row['source_annotation_file'],source_annotation_row=row['source_annotation_row'],source_annotation_sha256=row['source_annotation_sha256'],source_archive=video['archive_path'],source_archive_member=video['archive_member'],source_sha256=source_hash.hexdigest(),max_source_sample_time_error_s=exported['max_source_sample_time_error_s'],temporal_padding_required=frames<64,**verified))
   print('CLIP',key,len(clips),'class',row['label'],'frames',frames,flush=True)
  result=dict(key=key,signature=signature,source_sha256=source_hash.hexdigest(),source_duration_s=duration,source_fps=fps,clips=clips,exclusions=exclusions,status='decoded_outputs_verified',source_full_decode_performed=timing_audit is not None,source_timing_audit=timing_audit)
  write(done,result);return result
 finally:
  for p in [raw,temp]:
   if p.exists():
    if p.resolve().parent!=scratch.resolve():raise RuntimeError('Unexpected cleanup target')
    p.unlink()
def publish(protocol,videos,media_root):
 completed=[read(ROOT/'completed'/f'{v["key"]}.json') for v in videos if (ROOT/'completed'/f'{v["key"]}.json').exists()]
 clips=[r for x in completed for r in x['clips']];exclusions=[r for x in completed for r in x['exclusions']]
 status='ready_for_frozen_checkpoint_evaluation' if len(completed)==len(videos) else 'preparation_in_progress'
 progress=dict(status=status,completed_source_videos=len(completed),total_source_videos=len(videos),verified_test_camera_clips=len(clips),prepared_drivers=sorted({r['driver'] for r in clips}),planned_drivers=protocol['test_drivers'],excluded_annotation_rows=len(exclusions),class_counts=dict(collections.Counter(r['label'] for r in clips)),prepared_bytes=sum(r['size_bytes'] for r in clips),media_root=str(media_root),test_metrics_computed=False)
 write(ROOT/'progress.json',progress)
 if status=='ready_for_frozen_checkpoint_evaluation':
  assert {r['driver'] for r in clips}==set(protocol['test_drivers'])
  assert {r['label'] for r in clips}==set(range(16))
  assert not {r['driver'] for r in clips}&set(protocol['training_drivers']+protocol['validation_drivers'])
  assert len({r['sample_id'] for r in clips})==len(clips)
  for r in clips:
   if sha(media_root/r['path'])!=r['sha256']:raise ValueError('Final clip checksum mismatch')
  index=''.join(json.dumps(r)+'\n' for r in clips)
  (media_root/'test_index.jsonl').write_text(index,encoding='utf-8');(ROOT/'test_index.jsonl').write_text(index,encoding='utf-8')
  write(media_root/'protocol_lock.json',protocol);write(media_root/'annotation_exclusions.json',exclusions)
  shutil.copy2('F:/M2DAR_preprocessed/classes.json',media_root/'classes.json')
  if (ROOT/'fps_investigation.json').exists():shutil.copy2(ROOT/'fps_investigation.json',media_root/'source_timing_audit.json')
  ready=dict(progress,protocol_sha256=stable(protocol),index_sha256=sha(media_root/'test_index.jsonl'),checkpoint_sha256=CHECKPOINT,leakage_check_passed=True,all_output_clips_fully_decoded=True,selection_uses_model_predictions=False,three_camera_triplets_available=False)
  write(media_root/'READY.json',ready);write(ROOT/'READY.json',ready)
 return progress
def main():
 p=argparse.ArgumentParser();p.add_argument('--media-root',type=Path,default=Path('E:/IVP_GROUP16/M2DAR_testset'));p.add_argument('--freeze-only',action='store_true');p.add_argument('--limit-videos',type=int);a=p.parse_args();a.media_root.mkdir(parents=True,exist_ok=True)
 protocol,videos=freeze(a.media_root)
 if a.freeze_only:print(json.dumps(read(ROOT/'selection_summary.json'),indent=2));return
 spec=importlib.util.spec_from_file_location('locked_exporter','F:/A1_preprocessed/export_clips.py');exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter);exporter.OUT=a.media_root
 scratch=ROOT/'.staging';scratch.mkdir(exist_ok=True);started=time.time()
 for v in videos[:a.limit_videos] if a.limit_videos else videos:
  prepare_video(v,protocol,a.media_root,scratch,exporter);progress=publish(protocol,videos,a.media_root);print('PROGRESS',json.dumps(progress),'elapsed_s',round(time.time()-started,1),flush=True)
 final=publish(protocol,videos,a.media_root)
 if final['status']=='ready_for_frozen_checkpoint_evaluation' and (ROOT/'last_error.json').exists():
  prior=read(ROOT/'last_error.json');prior.update(resolved=True,resolution='Completed after source-byte-bound full decoded timing audit and PTS-based export; frozen test protocol unchanged.');write(ROOT/'last_error.json',prior)
 print(json.dumps(final,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(ROOT/'last_error.json',dict(error=str(e),time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ready=False))
  progress=read(ROOT/'progress.json') if (ROOT/'progress.json').exists() else {}
  progress.update(status='preparation_stopped',error=str(e));write(ROOT/'progress.json',progress)
  traceback.print_exc();raise SystemExit(1)
