from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
here=Path(r'F:\M2DAR_preprocessing')
stages=[[root/'test_alignment.py'],[root/'extract_motion.py','--workers','6','--device','d3d11va'],[here/'refresh_decoded_inventory.py',root],[root/'audit_labels.py'],[root/'fit_alignment.py'],[root/'make_clip_plan.py'],[root/'export_clips.py','--workers','4'],[root/'validate_exports.py','--workers','4','--publish'],[root/'verify_training.py'],[root/'create_previews.py'],[root/'finalize_package.py']]
for args in stages:subprocess.run([sys.executable,*map(str,args)],cwd=root,check=True)
