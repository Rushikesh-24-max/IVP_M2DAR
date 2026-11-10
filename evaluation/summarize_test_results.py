"""Independently recompute test metrics and driver-cluster bootstrap intervals."""
from pathlib import Path
import argparse,collections,csv,json
import numpy as np
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def metric(cm,classes=range(16)):
 cm=np.asarray(cm,dtype=float);tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);predicted=cm.sum(-2)
 precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0);recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0);f1=np.divide(2*tp,support+predicted,out=np.zeros_like(tp),where=support+predicted>0)
 return dict(accuracy=tp.sum(-1)/cm.sum(axis=(-2,-1)),macro_f1=f1[...,list(classes)].mean(-1),macro_precision=precision[...,list(classes)].mean(-1),macro_recall=recall[...,list(classes)].mean(-1),per_class_precision=precision,per_class_recall=recall,per_class_f1=f1,support=support)
def calculate(records,seed=42,replicates=2000):
 drivers=sorted({r['driver'] for r in records});matrices=np.zeros((len(drivers),16,16),dtype=np.int64);lookup={d:i for i,d in enumerate(drivers)};top5=0
 for r in records:
  p=np.array(r['probabilities']);assert p.shape==(16,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<.001
  assert type(r['label']) is int and 0<=r['label']<16
  matrices[lookup[r['driver']],r['label'],int(p.argmax())]+=1;top5+=int(r['label'] in np.argsort(-p,kind='stable')[:5])
 cm=matrices.sum(0);m=metric(cm);supported=np.flatnonzero(cm.sum(1)>0).tolist()
 counts=np.random.default_rng(seed).multinomial(len(drivers),np.full(len(drivers),1/len(drivers)),size=replicates)
 draws=np.einsum('bd,dij->bij',counts,matrices);sample=metric(draws);supported_draws=metric(draws,supported)
 result={k:float(m[k]) for k in ['accuracy','macro_f1','macro_precision','macro_recall']};result.update(samples=len(records),correct=int(np.trace(cm)),incorrect=int(len(records)-np.trace(cm)),top5_accuracy=top5/len(records),balanced_accuracy=float(m['macro_recall']),macro_f1_supported_classes=float(metric(cm,supported)['macro_f1']),supported_classes=supported,driver_count=len(drivers),confusion_matrix=cm.tolist(),bootstrap=dict(unit='driver',replicates=replicates,seed=seed,method='percentile 95% interval, resampling entire driver clusters with replacement',accuracy_95ci=np.quantile(sample['accuracy'],[.025,.975]).tolist(),macro_f1_all16_95ci=np.quantile(sample['macro_f1'],[.025,.975]).tolist(),macro_f1_supported_classes_95ci=np.quantile(supported_draws['macro_f1'],[.025,.975]).tolist()))
 result['per_class']=[dict(class_id=i,support=int(m['support'][i]),correct=int(cm[i,i]),precision=float(m['per_class_precision'][i]),recall=float(m['per_class_recall'][i]),f1=float(m['per_class_f1'][i])) for i in range(16)]
 return result
def write_csv(path,data):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 final=read(a.results/'final_metrics.json');assert final['status']=='completed' and not final['model_training_performed']
 primary_rows=rows(a.results/'primary_all_camera_clips/predictions.jsonl')
 frozen=rows(Path('F:/M2DAR_testset/test_index.jsonl'));truth={r['sample_id']:r for r in frozen}
 assert len(primary_rows)==len(truth)==len({r['sample_id'] for r in primary_rows})
 for r in primary_rows:
  reference=truth[r['sample_id']]
  for key in ['label','driver','session','view','path','sha256']:assert r[key]==reference[key]
 primary=calculate(primary_rows)
 for key in ['accuracy','macro_f1','balanced_accuracy','top5_accuracy']:assert abs(primary[key]-final['primary_metrics']['single_view_clips'][key])<1e-12
 assert primary['samples']==1748 and primary['driver_count']==19
 cameras={v:calculate([r for r in primary_rows if r['view']==v]) for v in ['dashboard','rearview','right_side_window']}
 for view,m in cameras.items():
  for key in ['accuracy','macro_f1','balanced_accuracy','top5_accuracy']:assert abs(m[key]-final['primary_metrics']['by_camera'][view][key])<1e-12
 classes=read(Path('E:/IVP_GROUP16/M2DAR_testset/classes.json'))
 for x in primary['per_class']:x['class_name']=classes[str(x['class_id'])]
 write_csv(a.out/'primary_per_class.csv',primary['per_class'])
 fused=None;selected_single=None;selected_cameras=None
 if final['verified_three_camera_subset_metrics'] is not None:
  selected_rows=rows(a.results/'verified_three_camera_subset/predictions.jsonl');fused_rows=rows(a.results/'verified_three_camera_subset/fused_predictions.jsonl')
  groups=rows(Path('F:/M2DAR_testset/sync/synchronized_test_groups.jsonl'));group_truth={g['group_id']:g for g in groups}
  assert len(fused_rows)==204==len(group_truth)==len({r['group_id'] for r in fused_rows})
  assert len(selected_rows)==612
  selected_groups=collections.defaultdict(list)
  for r in selected_rows:
   g=group_truth[r['group_id']];spec=g['views'][r['view']]
   assert r['driver']==g['driver'] and r['label']==g['label']
   for key in ['sample_id','path','sha256','start_frame','end_frame']:assert r[key]==spec[key]
   selected_groups[r['group_id']].append(r)
  for r in fused_rows:
   members=selected_groups[r['group_id']]
   assert len(members)==3 and {x['view'] for x in members}==set(cameras)
   assert r['driver']==group_truth[r['group_id']]['driver'] and r['label']==group_truth[r['group_id']]['label']
   assert np.allclose(r['probabilities'],np.mean([x['probabilities'] for x in members],axis=0),rtol=0,atol=1e-12)
  selected_single=calculate(selected_rows);selected_cameras={v:calculate([r for r in selected_rows if r['view']==v]) for v in cameras}
  for key in ['accuracy','macro_f1','balanced_accuracy','top5_accuracy']:assert abs(selected_single[key]-final['verified_three_camera_subset_metrics']['single_view_clips_same_subset'][key])<1e-12
  fused=calculate(fused_rows)
  for key in ['accuracy','macro_f1','balanced_accuracy','top5_accuracy']:assert abs(fused[key]-final['verified_three_camera_subset_metrics']['equal_weight_three_camera_classification'][key])<1e-12
  for x in fused['per_class']:x['class_name']=classes[str(x['class_id'])]
  write_csv(a.out/'verified_fusion_per_class.csv',fused['per_class'])
 report=dict(checkpoint_sha256=final['checkpoint_sha256'],primary_independent_test=primary,by_camera=cameras,verified_synchronized_subset=fused,synchronized_subset_individual_views=selected_single,synchronized_subset_by_camera=selected_cameras,synchronization=final['synchronization'],independent_numeric_recomputation_passed=True,prediction_membership_and_labels_reconciled=True,three_camera_averaging_recomputed=True,model_selected_using_test=False,limitations=['A1 held-out-driver classification, not A2 ground-truth accuracy or temporal localization performance.','Primary per-camera clips use each camera supplied action interval; previous validation used common aligned intervals, so clip selection differs.','Synchronized fusion covers only informative groups passing source and post-crop motion checks; report coverage and missing classes.','Synchronization is visually estimated; no hardware clock reference.','Intervals quantify sampling uncertainty across the 19 supplied driver IDs within this dataset, not guaranteed real-world performance.'])
 (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 lines=['# Frozen epoch-170 independent test results','',f'Checkpoint SHA256: `{final["checkpoint_sha256"]}`. No training or checkpoint selection used this test set.','', '## Primary test: all 1,748 per-camera clips from 19 unseen drivers','','| Metric | Result |','|---|---:|']
 for k,label in [('accuracy','Accuracy'),('macro_f1','Macro-F1'),('macro_precision','Macro-precision'),('macro_recall','Macro-recall / balanced accuracy'),('top5_accuracy','Top-5 accuracy')]:lines.append(f'| {label} | {primary[k]:.2%} |')
 for k,label in [('accuracy_95ci','Accuracy'),('macro_f1_all16_95ci','Macro-F1')]:
  lo,hi=primary['bootstrap'][k];lines.append(f'\n{label} driver-cluster bootstrap 95% interval: {lo:.2%}–{hi:.2%}.')
 lines+=['','## Per-camera primary test','','| Camera | Clips | Accuracy | Macro-F1 |','|---|---:|---:|---:|']
 for v,m in cameras.items():lines.append(f'| {v} | {m["samples"]} | {m["accuracy"]:.2%} | {m["macro_f1"]:.2%} |')
 sync=final['synchronization'];lines+=['','## Synchronization assessment','',f'Original source videos checked: {sync["source_videos_checked"]}; source recording sessions: {sync["source_recordings"]}. Candidate aligned groups: {sync["candidate_aligned_groups"]}; verified: {sync["verified_groups"]}; excluded or inconclusive: {sync["excluded_or_inconclusive_groups"]}.']
 if fused:
  lines += ['',f'Fusion subset: **{fused["accuracy"]:.2%} accuracy**, **{fused["macro_f1_supported_classes"]:.2%} macro-F1 over supported classes**, {fused["samples"]} verified groups from {fused["driver_count"]} drivers. Supported class IDs: {fused["supported_classes"]}.',f'All-16-class macro-F1 (zero for unsupported classes): {fused["macro_f1"]:.2%}. Do not present the subset score as performance on all 1,748 clips.']
  lines+=['',f'The same synchronized subset before fusion has {selected_single["samples"]} individual camera views: {selected_single["accuracy"]:.2%} accuracy and {selected_single["macro_f1"]:.2%} macro-F1. This is the matching population for assessing three-camera averaging.','', '| Metric | Three-camera fusion |','|---|---:|']
  for k,label in [('macro_precision','Macro-precision'),('macro_recall','Macro-recall / balanced accuracy'),('top5_accuracy','Top-5 accuracy')]:lines.append(f'| {label} | {fused[k]:.2%} |')
  for k,label in [('accuracy_95ci','Accuracy'),('macro_f1_all16_95ci','Macro-F1')]:
   lo,hi=fused['bootstrap'][k];lines.append(f'\n{label} driver-cluster bootstrap 95% interval: {lo:.2%}–{hi:.2%}.')
 else:lines+=['','No verified fusion subset was available; no fusion accuracy is reported.']
 lines+=['','## Interpretation','',*['- '+x for x in report['limitations']],'','The epoch-170 checkpoint was selected using the earlier eight-driver A1 validation set. This separate test set had 19 unseen drivers. The dataset preparation, label mapping and synchronization policy were fixed without reading these model predictions. Keep this test set for final reporting; if future model changes are tuned to its outcomes, use a new untouched test set.','']
 (a.out/'README.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({k:primary[k] for k in ['samples','accuracy','macro_f1','macro_precision','macro_recall','top5_accuracy']},indent=2))
if __name__=='__main__':main()
