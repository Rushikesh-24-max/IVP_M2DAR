"""Disposable GPU worker. One file, one viewpoint, immutable checkpoint."""
from __future__ import annotations
from pathlib import Path
import argparse,bisect,hashlib,json,math,sys,time
from types import SimpleNamespace
from protocol import CHECKPOINT,SOURCE_LOCK,DATASET,FileLock,atomic,read,sha256,vector
from model_runtime import config,load_model,readiness

def run(job_file,training_root,bundle,checkpoint):
 folder=job_file.parent;j=read(job_file);probe=read(folder/'probe.json');video=folder/'video.mp4'
 if len(j['views'])!=1 or sha256(video)!=probe['sha256']:raise ValueError('Single-video source integrity check failed')
 view=j['views'][0];duration=probe['duration_s'];pts=probe['pts_s']
 if not math.isfinite(duration) or len(pts)!=probe['frames']:raise ValueError('Invalid validated media timing')
 with FileLock(Path(training_root)/'.m2dar-single-video-gpu1.lock'):
  cfg,classes=config(training_root,bundle,checkpoint)
  from common import apply_gpu_memory_limit
  import torch,numpy as np
  apply_gpu_memory_limit(.20);model=load_model(cfg,checkpoint)
  def progress(done,total,message):atomic(folder/'progress.json',dict(progress=100*done/max(1,total),completed_windows=done,total_windows=total,message=message))
  windows=[];input_hash=None
  with torch.inference_mode():
   if j['inference_mode']=='clip_center':
    # Reuse the exact independent-test val loader, with a private placeholder
    # label solely to satisfy its CSV schema. It is never treated as ground truth.
    if not probe['constant_frame_rate']:raise ValueError('clip_center comparison requires constant-frame-rate input; use sliding_window for variable source timing')
    from slowfast.datasets import loader
    compat=folder/'loader_inputs';compat.mkdir(exist_ok=True)
    (compat/'val.csv').write_text('video.mp4'+cfg.DATA.PATH_LABEL_SEPARATOR+'0\n',encoding='utf-8')
    cfg.DATA.PATH_TO_DATA_DIR=str(compat);cfg.DATA.PATH_PREFIX=str(folder);cfg.DATA.VAL_SKIP=1;cfg.DATA_LOADER.NUM_WORKERS=0;cfg.TRAIN.BATCH_SIZE=1;cfg.TEST.BATCH_SIZE=1
    batches=loader.construct_loader(cfg,'val');progress(0,1,'Evaluating the centre window')
    for inputs,_,_,_ in batches:
     input_hash=hashlib.sha256(inputs[0].numpy().tobytes()).hexdigest()
     p=model([x.cuda(non_blocking=True) for x in inputs]).float().cpu().tolist()[0];vector(p)
     windows.append(dict(start_s=0.,end_s=duration,probabilities=p,scope='clip_level_center_window_not_temporal_localization'))
    if len(windows)!=1:raise ValueError('Expected exactly one centre-window prediction')
    progress(1,1,'Centre-window prediction completed')
   else:
    sys.path.insert(0,str(Path(training_root)/'vendor/M2DAR/tools/inference'))
    from module_wrapper import ActionProposalFromVideoTemporalDataset
    import decord
    reader=decord.VideoReader(str(video),num_threads=1)
    if len(reader)!=len(pts):raise ValueError('Independent decoders disagree on source frame count')
    args=SimpleNamespace(frame_format='rgb',video_decoder='decord',frame_length=16,frame_stride=4,frame_size=448,model_type='mvit',roi_x1=0.,roi_y1=0.,roi_x2=1.,roi_y2=1.,video_fps=-1.,target_fps=30.,proposal_stride=16,proposal_length=64)
    timed=not probe['constant_frame_rate'] or reader.get_avg_fps()<2
    if timed:args.video_fps=30.
    class TimedDataset(ActionProposalFromVideoTemporalDataset):
     def _get_frame_idxs_uniform(self,start,end,num_frames):
      # Source-time floor selection; repeat observed edge frames as upstream.
      targets=torch.linspace(start/30.,end/30.,num_frames).tolist()
      return [min(len(pts)-1,max(0,bisect.bisect_right(pts,t)-1)) for t in targets]
    dataset=(TimedDataset if timed else ActionProposalFromVideoTemporalDataset)(str(video),args);dataset.video_cap=reader
    if timed:dataset.prop_data=[[dataset.roi_tlbr,start,start+64] for start in range(0,math.ceil(duration*30),16)]
    total=len(dataset);progress(0,total,'Running single-camera prediction windows')
    for i in range(total):
     _,a,b=dataset.prop_data[i]
     start=a/30 if timed else pts[min(a,len(pts)-1)]
     end=min(b/30,duration) if timed else (pts[b] if b<len(pts) else duration)
     if any(start<g['end_s'] and end>g['start_s'] for g in probe.get('unsupported_intervals',[])):
      progress(i+1,total,f'Skipped unsupported source timing at window {i+1}/{total}');continue
     inputs,a,b=dataset[i];p=model([torch.from_numpy(x).unsqueeze(0).cuda() for x in inputs]).float().cpu().numpy()[0].tolist();vector(p)
     start=a/30 if timed else pts[min(a,len(pts)-1)]
     end=min(b/30,duration) if timed else (pts[b] if b<len(pts) else duration)
     if start<end:windows.append(dict(start_s=start,end_s=end,probabilities=p))
     progress(i+1,total,f'Analyzed {i+1}/{total} windows')
    dataset.video_cap=None;del dataset,reader
  result=dict(schema_version=1,id=j['id'],title=j['title'],status='completed',source='GPU inference',checkpoint_sha256=CHECKPOINT,source_lock_sha256=SOURCE_LOCK,dataset_fingerprint=DATASET,inference_mode=j['inference_mode'],completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=probe['sha256'],duration_s=duration,reference_start_s=0.,views={view:dict(duration_s=duration,offset_s=0,windows=windows)},classes=classes,ground_truth_available=False,preprocessing_version='m2dar-single-video-v1',input_tensor_sha256=input_hash,timing_policy=probe['timing_policy'],method='One deterministic centre-window classification' if j['inference_mode']=='clip_center' else 'Single-camera scores; average covering temporal windows at the playhead. No cross-camera synchronization or fusion.',limitation='Clip classification metrics do not measure full-video localization. No ground-truth label is inferred for an upload.')
  if not windows:raise ValueError('No valid prediction windows')
  atomic(folder/'result.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--readiness-out',type=Path);a=p.parse_args()
 if a.check:atomic(a.readiness_out,readiness(a.training_root,a.bundle,a.checkpoint))
 elif a.job:run(a.job,a.training_root,a.bundle,a.checkpoint)
 else:p.error('--job or --check is required')
