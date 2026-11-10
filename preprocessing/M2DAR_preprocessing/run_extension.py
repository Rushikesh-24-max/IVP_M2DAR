"""Resume-safe sequential processing of the requested additional dataset parts."""
from pathlib import Path
import sys,subprocess,json,time,shutil,argparse,traceback
HERE=Path(__file__).resolve().parent;START=time.time()
PARTS=['A1_2','A1_3','A1_4']

def stage(root,args):
    name=Path(args[0]).name
    if shutil.disk_usage(root).free<4_000_000_000:raise RuntimeError('Less than 4 GB free: stopped before '+name)
    state=dict(package=root.name,stage=name,args=list(map(str,args)),status='running',elapsed_s=time.time()-START)
    (HERE/'progress.json').write_text(json.dumps(state,indent=2));print('STAGE',root.name,name,flush=True)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    with (logs/(name+'.log')).open('a',encoding='utf-8') as log:
        log.write('\nRUN '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');log.flush()
        subprocess.run([sys.executable,*map(str,args)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    state.update(status='complete',elapsed_s=time.time()-START);(HERE/'progress.json').write_text(json.dumps(state,indent=2))
    print('DONE',root.name,name,'elapsed',round(time.time()-START,1),flush=True)

def run():
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');p.add_argument('--skip-a2',action='store_true');args=p.parse_args()
    for part in PARTS:
        root=Path('F:/')/(part+'_preprocessed')
        stages=[[root/'test_alignment.py'],[root/'extract_motion.py','--workers','6']+([] if args.cpu else ['--device','d3d11va']),
            [HERE/'refresh_decoded_inventory.py',root],[root/'audit_labels.py'],[root/'fit_alignment.py'],
            [root/'make_clip_plan.py'],[root/'export_clips.py','--workers','4'],
            [root/'validate_exports.py','--workers','4','--publish'],[root/'verify_training.py'],[root/'create_previews.py'],[root/'finalize_package.py']]
        for command in stages:stage(root,command)
    if not args.skip_a2:
        root=Path(r'F:\A2_preprocessed')
        for command in [[root/'test_alignment.py'],[root/'extract_motion.py','--workers','6']+([] if args.cpu else ['--device','d3d11va']),
            [HERE/'refresh_decoded_inventory.py',root],[root/'fit_alignment.py'],[HERE/'prepare_a2.py'],[HERE/'verify_a2.py']]:stage(root,command)
    stage(HERE,[HERE/'combine_training.py'])
    (HERE/'progress.json').write_text(json.dumps(dict(status='complete',elapsed_s=time.time()-START),indent=2))
    print('EXTENSION COMPLETE',round(time.time()-START,1),flush=True)

if __name__=='__main__':
    try:run()
    except Exception as e:
        p=HERE/'progress.json';state=json.loads(p.read_text()) if p.exists() else {}
        state.update(status='failed',error=repr(e));p.write_text(json.dumps(state,indent=2));traceback.print_exc();sys.exit(1)
