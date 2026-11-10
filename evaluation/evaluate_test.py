"""Evaluate the frozen independent test set; never select checkpoints or train."""
from __future__ import annotations
import argparse,collections,hashlib,json,sys,time
from pathlib import Path
EXPECTED_CHECKPOINT='f2a254dca49cd5156b344d3abff40e21061bb3fc4e605ba1728925d2d3db3115'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def stable(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_loader_csv(path,rows,separator,path_key='path'):
 if not isinstance(separator,str) or not separator or '\n' in separator or '\r' in separator:raise ValueError('Invalid configured path/label separator')
 lines=[]
 for row in rows:
  value=row[path_key]
  if separator in value or '\n' in value or '\r' in value:raise ValueError('Unsafe loader path')
  lines.append(f'{value}{separator}{row["label"]}\n')
 Path(path).write_text(''.join(lines),encoding='utf-8')
def verify_test(root,checkpoint=None,check_media=True):
 root=Path(root).resolve();ready=read(root/'READY.json');protocol=read(root/'protocol_lock.json')
 if ready.get('status')!='ready_for_frozen_checkpoint_evaluation':raise ValueError('Test preparation is incomplete')
 if ready['protocol_sha256']!=stable(protocol):raise ValueError('Test protocol changed')
 if digest(root/'test_index.jsonl')!=ready['index_sha256']:raise ValueError('Frozen test index changed')
 if protocol['checkpoint_sha256']!=EXPECTED_CHECKPOINT or ready['checkpoint_sha256']!=EXPECTED_CHECKPOINT:raise ValueError('Unexpected checkpoint in protocol')
 if checkpoint is not None and digest(checkpoint)!=EXPECTED_CHECKPOINT:raise ValueError('Use the fixed epoch-170 checkpoint. Do not choose another checkpoint on this test set.')
 rows=[json.loads(s) for s in (root/'test_index.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
 if len(rows)!=ready['verified_test_camera_clips'] or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Wrong test sample count or duplicate identifiers')
 drivers={r['driver'] for r in rows}
 if drivers!=set(protocol['test_drivers']) or drivers&set(protocol['training_drivers']+protocol['validation_drivers']):raise ValueError('Test driver leakage or changed membership')
 if {r['label'] for r in rows}!=set(range(16)):raise ValueError('Not all classes are represented')
 for row in rows:
  if row['split']!='test' or type(row['label']) is not int or not 0<=row['label']<16:raise ValueError('Invalid test row')
  file=(root/row['path']).resolve()
  if not file.is_relative_to(root) or not file.is_file():raise ValueError('Missing or unsafe test media path')
  if check_media and (file.stat().st_size!=row['size_bytes'] or digest(file)!=row['sha256']):raise ValueError('Test media changed: '+row['sample_id'])
 return protocol,rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--test-root',type=Path,required=True);p.add_argument('--training-root',type=Path,default=Path.home()/'m2dar-training');p.add_argument('--bundle',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 checkpoint=a.checkpoint or a.training_root/'runs/m2dar_s_40drivers_b1_amp_sdpa_seed42/best.pyth'
 protocol,rows=verify_test(a.test_root,None if a.verify_only else checkpoint)
 if a.verify_only:print(f'Frozen test verified: {len(rows)} camera clips, {len(protocol["test_drivers"])} unseen drivers. No model was run.');return
 out=a.out.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory; preserve existing test outcomes')
 sys.path.insert(0,str(a.training_root/'scripts'))
 from common import read_cfg,load_torch_checkpoint,require_shapes,require_gpu,seed_runtime,apply_gpu_memory_limit,verify_bundle
 from metrics import classification_metrics
 require_gpu();apply_gpu_memory_limit(.20);seed_runtime(42)
 bundle=a.bundle or a.training_root/'prepared/server_b1_amp_sdpa';bundle_meta=verify_bundle(bundle,check_files=False)
 run=read(checkpoint.parent/'run.json')
 if bundle_meta['dataset_fingerprint']!=run['dataset_fingerprint']:raise ValueError('Original training bundle does not match this checkpoint')
 # Reject leakage against the actual loaded training bundle as well as the frozen local split.
 actual_used={r['driver'] for name in ['train_index.jsonl','val_index.jsonl'] for r in [json.loads(x) for x in (bundle/name).read_text().splitlines() if x.strip()]}
 if actual_used & set(protocol['test_drivers']):raise ValueError('Loaded checkpoint bundle exposes test drivers')
 cfg=read_cfg(bundle/'baseline.yaml');out.mkdir(parents=True,exist_ok=True);compat=out/'loader_inputs';compat.mkdir()
 # The upstream val branch provides the SAME one-center-window protocol used for
 # previous validation. This private compatibility list is still TEST data.
 write_loader_csv(compat/'val.csv',rows,cfg.DATA.PATH_LABEL_SEPARATOR)
 cfg.DATA.PATH_TO_DATA_DIR=str(compat);cfg.DATA.PATH_PREFIX=str(a.test_root.resolve());cfg.DATA.VAL_SKIP=1
 cfg.DATA_LOADER.NUM_WORKERS=0;cfg.TEST.BATCH_SIZE=1;cfg.TRAIN.BATCH_SIZE=1
 import torch
 from slowfast.models import build_model
 from slowfast.datasets import loader
 state=load_torch_checkpoint(checkpoint)
 if 'preparation' in state:raise ValueError('Initializer cannot be used as a trained checkpoint')
 model=build_model(cfg);require_shapes(model.state_dict(),state['model_state']);model.load_state_dict(state['model_state'],strict=True);model.eval();del state
 predictions={};began=time.time()
 with torch.inference_mode():
  for inputs,labels,ids,_ in loader.construct_loader(cfg,'val'):
   values=model([x.cuda(non_blocking=True) for x in inputs]).float().cpu().tolist()
   for i,c,prob in zip(ids.tolist(),labels.tolist(),values):
    if i in predictions or rows[i]['label']!=c:raise ValueError('Repeated or mislabeled test sample')
    predictions[i]=dict(rows[i],probabilities=prob)
   if len(predictions)%25==0:print(f'Test {len(predictions)}/{len(rows)}; elapsed {time.time()-began:.1f}s',flush=True)
 if set(predictions)!=set(range(len(rows))):raise ValueError('Incomplete test predictions')
 ordered=[predictions[i] for i in range(len(rows))]
 def score(rr):return classification_metrics([r['label'] for r in rr],[r['probabilities'] for r in rr])
 result=dict(split='independent_test',single_view_clips=score(ordered),by_camera={v:score([r for r in ordered if r['view']==v]) for v in ['dashboard','rearview','right_side_window']},by_driver={d:score([r for r in ordered if r['driver']==d]) for d in protocol['test_drivers']},checkpoint_sha256=EXPECTED_CHECKPOINT,test_protocol_sha256=stable(protocol),test_index_sha256=digest(a.test_root/'test_index.jsonl'),original_training_dataset_fingerprint=bundle_meta['dataset_fingerprint'],primary_metric='pooled single-camera macro-F1',protocol='One deterministic center temporal window and spatial crop per labeled per-camera clip. No three-camera fusion without verified synchronization.',checkpoint_selected_using_this_test=False,model_training_performed=False)
 (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ordered),encoding='utf-8');(out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result['single_view_clips'].items() if k not in ['confusion_matrix','per_class']},indent=2))
if __name__=='__main__':main()
