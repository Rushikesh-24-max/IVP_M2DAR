"""Reproduce the authorized A1-only preparation stages using local source data."""
from pathlib import Path
import sys,subprocess,argparse
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',action='store_true');args=p.parse_args()
    stages=[['audit_labels.py'],['extract_motion.py','--workers','6']+([] if args.cpu else ['--device','d3d11va']),
            ['test_alignment.py'],['fit_alignment.py'],['make_clip_plan.py'],['export_clips.py','--workers','4'],
            ['validate_exports.py','--workers','4','--publish'],['verify_training.py'],['create_previews.py'],['finalize_package.py']]
    for stage in stages:
        print('STAGE',stage[0],flush=True)
        subprocess.run([sys.executable,str(ROOT/stage[0]),*stage[1:]],cwd=ROOT,check=True)
