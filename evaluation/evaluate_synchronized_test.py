"""Evaluate aligned virtual frame ranges with the unchanged validation loader."""
from pathlib import Path
import argparse,collections,json,sys,time
from evaluate_test import verify_test,read,digest,stable,write_loader_csv,EXPECTED_CHECKPOINT
VIEWS=('dashboard','rearview','right_side_window')
class FrameRangeReader:
 """Zero-copy index translation over a verified prepared camera clip."""
 def __init__(self,reader,start,end):
  if type(start) is not int or type(end) is not int or not 0<=start<end<=len(reader):raise ValueError('Invalid aligned frame range')
  self.reader=reader;self.start=start;self.end=end
 def __len__(self):return self.end-self.start
 def get_avg_fps(self):return self.reader.get_avg_fps()
 def get_batch(self,indices):
  indices=[int(i) for i in indices]
  if any(i<0 or i>=len(self) for i in indices):raise ValueError('Requested frame outside aligned range')
  return self.reader.get_batch([self.start+i for i in indices])
def verify_sync(root,sync_root,checkpoint=None):
 protocol,rows=verify_test(root,checkpoint)
 s=read(sync_root/'READY_SYNC.json');policy=read(sync_root/'sync_protocol.json')
 if s['status']!='synchronization_assessment_complete' or s['model_predictions_used']:raise ValueError('Synchronization assessment is incomplete or not model-blind')
 if stable(policy)!=s['protocol_sha256'] or policy['primary_test_protocol_sha256']!=stable(protocol) or s['primary_index_sha256']!=digest(root/'test_index.jsonl'):raise ValueError('Synchronization protocol/test binding changed')
 for name,key in [('synchronized_test_groups.jsonl','groups_sha256'),('full_source_motion_audit.json','source_motion_audit_sha256'),('group_checks.json','group_checks_sha256'),('alignment_estimates.json','alignment_sha256')]:
  if digest(sync_root/name)!=s[key]:raise ValueError('Synchronization artifact changed: '+name)
 groups=[json.loads(x) for x in (sync_root/'synchronized_test_groups.jsonl').read_text().splitlines() if x.strip()]
 if len(groups)!=s['verified_groups'] or len({g['group_id'] for g in groups})!=len(groups):raise ValueError('Synchronization group count mismatch')
 byid={r['sample_id']:r for r in rows}
 for g in groups:
  if not g['verification']['verified'] or set(g['views'])!=set(VIEWS) or g['frame_count']<16:raise ValueError('Unverified synchronization group')
  for v in VIEWS:
   spec=g['views'][v];r=byid[spec['sample_id']];check=g['verification']['post_crop_alignment'].get(v)
   if v!='dashboard' and (check['status']!='informative' or check.get('failed') or abs(check['residual_frames'])>3):raise ValueError('Residual alignment gate failed')
   if r['driver']!=g['driver'] or r['session']!=g['session'] or r['label']!=g['label'] or r['view']!=v or r['path']!=spec['path'] or r['sha256']!=spec['sha256']:raise ValueError('Mispaired camera group')
   if not 0<=spec['start_frame']<spec['end_frame']<=r['frames'] or spec['end_frame']-spec['start_frame']!=g['frame_count']:raise ValueError('Invalid synchronized crop')
 return protocol,groups,s
def main():
 p=argparse.ArgumentParser();p.add_argument('--test-root',type=Path,required=True);p.add_argument('--sync-root',type=Path,required=True);p.add_argument('--training-root',type=Path,default=Path.home()/'m2dar-training');p.add_argument('--bundle',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 root=a.test_root.resolve();sync=a.sync_root.resolve();ckpt=(a.checkpoint or a.training_root/'runs/m2dar_s_40drivers_b1_amp_sdpa_seed42/best.pyth').resolve()
 protocol,groups,syncready=verify_sync(root,sync,None if a.verify_only else ckpt)
 if a.verify_only:print('SYNC_TEST_VERIFIED',len(groups),'groups');return
 if not groups:raise ValueError('No synchronized groups passed verification; no fusion metric may be reported')
 if a.out.exists() and any(a.out.iterdir()):raise ValueError('Use a fresh output directory')
 sys.path.insert(0,str(a.training_root/'scripts'))
 from common import read_cfg,verify_bundle,read_jsonl,apply_gpu_memory_limit,seed_runtime,require_gpu,load_torch_checkpoint,require_shapes
 from metrics import classification_metrics
 bundle=a.bundle or a.training_root/'prepared/server_b1_amp_sdpa';original=verify_bundle(bundle,check_files=False)
 if original['dataset_fingerprint']!=read(ckpt.parent/'run.json')['dataset_fingerprint']:raise ValueError('Checkpoint bundle mismatch')
 used={r['driver'] for name in ['train_index.jsonl','val_index.jsonl'] for r in read_jsonl(bundle/name)}
 if used & set(protocol['test_drivers']):raise ValueError('Driver leakage into the checkpoint training bundle')
 require_gpu();apply_gpu_memory_limit(.20);seed_runtime(42);cfg=read_cfg(bundle/'baseline.yaml')
 if cfg.DATA.DECODING_BACKEND!='decord':raise ValueError('This range-view adapter requires the existing decord validation backend')
 import torch
 from slowfast.models import build_model
 from slowfast.datasets import loader,video_container
 state=load_torch_checkpoint(ckpt);model=build_model(cfg);require_shapes(model.state_dict(),state['model_state']);model.load_state_dict(state['model_state'],strict=True);model.eval();del state
 a.out.mkdir(parents=True);compat=a.out/'loader_inputs';compat.mkdir();cfg.DATA.PATH_TO_DATA_DIR=str(compat.resolve());cfg.DATA.PATH_PREFIX=str(root);cfg.DATA.VAL_SKIP=1;cfg.DATA_LOADER.NUM_WORKERS=0;cfg.TRAIN.BATCH_SIZE=1;cfg.TEST.BATCH_SIZE=1
 index=[];mapping={}
 for g in groups:
  for view in VIEWS:
   spec=g['views'][view];virtual=f'.virtual_sync/{g["group_id"]}_{view}.mp4';mapping[str(root/virtual)]=spec
   index.append(dict(group_id=g['group_id'],driver=g['driver'],session=g['session'],view=view,label=g['label'],virtual_path=virtual,**spec))
 write_loader_csv(compat/'val.csv',index,cfg.DATA.PATH_LABEL_SEPARATOR,'virtual_path')
 original_container=video_container.get_video_container
 def aligned_container(path,*args,**kwargs):
  if path not in mapping:raise ValueError('Unexpected virtual video path')
  spec=mapping[path];reader=original_container(str(root/spec['path']),*args,**kwargs)
  return FrameRangeReader(reader,spec['start_frame'],spec['end_frame'])
 video_container.get_video_container=aligned_container
 predictions={};started=time.time()
 try:
  with torch.inference_mode():
   for inputs,labels,ids,_ in loader.construct_loader(cfg,'val'):
    probs=model([x.cuda(non_blocking=True) for x in inputs]).float().cpu().tolist()
    for i,label,prob in zip(ids.tolist(),labels.tolist(),probs):
     if i in predictions or index[i]['label']!=label:raise ValueError('Invalid synchronized sample index')
     predictions[i]=dict(index[i],probabilities=prob)
    if len(predictions)%25==0:print('SYNC TEST',len(predictions),'/',len(index),'seconds',round(time.time()-started,1),flush=True)
 finally:video_container.get_video_container=original_container
 if set(predictions)!=set(range(len(index))):raise ValueError('Missing synchronized predictions')
 rows=[predictions[i] for i in range(len(index))];bygroup=collections.defaultdict(list)
 for r in rows:bygroup[r['group_id']].append(r)
 fused=[]
 for gid,rs in bygroup.items():
  if len(rs)!=3 or len({r['label'] for r in rs})!=1:raise ValueError('Invalid fusion group')
  fused.append(dict(group_id=gid,driver=rs[0]['driver'],label=rs[0]['label'],probabilities=[sum(r['probabilities'][c] for r in rs)/3 for c in range(16)]))
 def metrics(rr):
  m=classification_metrics([r['label'] for r in rr],[r['probabilities'] for r in rr]);supported=[r for r in m['per_class'] if r['support']]
  m['supported_classes']=[r['class_id'] for r in supported];m['macro_f1_over_supported_classes']=sum(r['f1'] for r in supported)/len(supported);return m
 result=dict(split='independent_test_synchronization_verified_subset',equal_weight_three_camera_classification=metrics(fused),single_view_clips_same_subset=metrics(rows),by_camera={v:metrics([r for r in rows if r['view']==v]) for v in VIEWS},checkpoint_sha256=EXPECTED_CHECKPOINT,primary_test_protocol_sha256=stable(protocol),sync_assessment=syncready,protocol='Exact integer frame-range views of prepared clips; unchanged deterministic center-window/crop validation decoder and transforms; no re-encoding or training.',limitation='Only source-supported groups with both informative independent post-crop checks and <=3-frame residual are fused. Subset coverage is reported; this is estimated visual synchronization, not hardware-clock ground truth.',checkpoint_selected_using_test=False)
 (a.out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');(a.out/'fused_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fused),encoding='utf-8');(a.out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print('SYNCHRONIZED_TEST_COMPLETED',len(fused),'triplets',flush=True)
if __name__=='__main__':main()
