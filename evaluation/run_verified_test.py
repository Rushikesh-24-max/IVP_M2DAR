"""Run the fixed-checkpoint primary test and verified fusion subset serially."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from evaluate_test import read,stable,EXPECTED_CHECKPOINT
from evaluate_synchronized_test import verify_sync
def main():
 p=argparse.ArgumentParser();p.add_argument('--test-root',type=Path,required=True);p.add_argument('--sync-root',type=Path,required=True);p.add_argument('--training-root',type=Path,default=Path.home()/'m2dar-training');p.add_argument('--out',type=Path,required=True);p.add_argument('--wait-for-ready',action='store_true');a=p.parse_args()
 while a.wait_for_ready and not (a.sync_root/'READY_TO_EVALUATE').exists():
  print('Waiting for completed synchronization assessment; no GPU model is running.',flush=True);time.sleep(30)
 checkpoint=a.training_root/'runs/m2dar_s_40drivers_b1_amp_sdpa_seed42/best.pyth'
 protocol,groups,ready=verify_sync(a.test_root.resolve(),a.sync_root.resolve(),checkpoint)
 a.out.mkdir(parents=True,exist_ok=True)
 lock=a.out/'.run.lock'
 try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:raise RuntimeError('Evaluation lock exists; do not launch a second run')
 os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  scripts=Path(__file__).resolve().parent;bundle=a.training_root/'prepared/server_b1_amp_sdpa'
  primary=a.out/'primary_all_camera_clips';secondary=a.out/'verified_three_camera_subset'
  command=[sys.executable,'-u',str(scripts/'evaluate_test.py'),'--test-root',str(a.test_root),'--training-root',str(a.training_root),'--bundle',str(bundle),'--checkpoint',str(checkpoint),'--out',str(primary)]
  print('SYNCHRONIZATION_GATE_PASSED. Starting fixed-checkpoint primary test.',flush=True)
  if not (primary/'metrics.json').exists():subprocess.run(command,cwd=a.training_root,check=True)
  primary_metrics=read(primary/'metrics.json')
  if primary_metrics['checkpoint_sha256']!=EXPECTED_CHECKPOINT or primary_metrics['test_protocol_sha256']!=stable(protocol) or primary_metrics['test_index_sha256']!=ready['primary_index_sha256']:raise ValueError('Existing primary metrics belong to another experiment')
  if groups:
   command=[sys.executable,'-u',str(scripts/'evaluate_synchronized_test.py'),'--test-root',str(a.test_root),'--sync-root',str(a.sync_root),'--training-root',str(a.training_root),'--bundle',str(bundle),'--checkpoint',str(checkpoint),'--out',str(secondary)]
   if not (secondary/'metrics.json').exists():subprocess.run(command,cwd=a.training_root,check=True)
   secondary_metrics=read(secondary/'metrics.json')
   if secondary_metrics['checkpoint_sha256']!=EXPECTED_CHECKPOINT or secondary_metrics['sync_assessment']['groups_sha256']!=ready['groups_sha256']:raise ValueError('Existing synchronized metrics belong to another experiment')
  result=dict(status='completed',checkpoint_sha256=EXPECTED_CHECKPOINT,synchronization=ready,primary_metrics=primary_metrics,verified_three_camera_subset_metrics=read(secondary/'metrics.json') if groups else None,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),model_training_performed=False)
  (a.out/'final_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print('INDEPENDENT_TEST_COMPLETE',flush=True)
 finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
