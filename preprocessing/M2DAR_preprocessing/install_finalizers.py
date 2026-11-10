"""Adapt only generated package documentation; keep original A1 results intact."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
template=Path(r'F:\A1_preprocessed\finalize_package.py').read_text()
old="Of {audit['annotation_rows']} local annotation rows, {audit['excluded_rows']} were excluded: eight outside available footage, three zero-duration rows, and three copies of a long conflicting annotation. The `16700/7` right-side video ends at 268 seconds. No missing camera footage or missing class labels were invented. Uncertain timing transitions, sustained disagreements and unsupported intervals were kept outside the prepared subset."
new="Of {audit['annotation_rows']} parsed local annotation rows, {audit['excluded_rows']} were excluded. Issues: {json.dumps(audit['issues'],sort_keys=True)}. See `annotation_issues.json` for affected source files and rows. No missing camera footage or missing class labels were invented. Uncertain timing transitions, sustained disagreements and unsupported intervals were kept outside the prepared subset.\n\nCamera-prefix corrections: {audit.get('camera_mapping_corrections',0)}. `annotation_corrections.json` records the inspected driver-47158 correction using the supplied Camera View column; action class IDs and times are unchanged."
assert old in template
for part in ['A1_2','A1_3','A1_4']:
    root=Path('F:/')/(part+'_preprocessed')
    text=template.replace(old,new).replace('A1_preprocessed',root.name).replace('A1_extracted only',part+' only')
    (root/'finalize_package.py').write_text(text)
    (root/'run_pipeline.py').write_text('''from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
here=Path(r'F:\\M2DAR_preprocessing')
stages=[[root/'test_alignment.py'],[root/'extract_motion.py','--workers','6','--device','d3d11va'],[here/'refresh_decoded_inventory.py',root],[root/'audit_labels.py'],[root/'fit_alignment.py'],[root/'make_clip_plan.py'],[root/'export_clips.py','--workers','4'],[root/'validate_exports.py','--workers','4','--publish'],[root/'verify_training.py'],[root/'create_previews.py'],[root/'finalize_package.py']]
for args in stages:subprocess.run([sys.executable,*map(str,args)],cwd=root,check=True)
''')
print('Installed package finalizers and resumable per-folder entry points.')
