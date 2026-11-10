"""Initialize additional A1 packages and the held-out, unlabelled A2 package."""
from pathlib import Path
import sys,json,shutil,hashlib,collections
BASE=Path(r'F:\A1_preprocessed');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE));sys.path.insert(0,r'F:\A1_analysis\tools')
from audit_labels import identify
import av

def initialize(raw,out,role):
    out.mkdir(exist_ok=True)
    paths=sorted(p for p in raw.rglob('*') if p.is_file() and p.suffix.lower()=='.mp4')
    rows=[];seen=set()
    for p in paths:
        driver,session,view=identify(p.name);key=(driver,session,view)
        if key in seen:raise ValueError('Duplicate source key: '+str(key))
        seen.add(key);st=p.stat()
        with av.open(str(p)) as c:
            s=c.streams.video[0]
            duration=float(s.duration*s.time_base) if s.duration is not None else c.duration/1e6
            row=dict(driver=driver,session=session,view=view,path=str(p),duration_s=duration,
                header_duration_s=duration,width=s.width,height=s.height,fps=float(s.average_rate),
                header_frames=s.frames,video_streams=len(c.streams.video),audio_streams=len(c.streams.audio),
                source_size_bytes=st.st_size,source_mtime_ns=st.st_mtime_ns)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for pos in [0,max(0,st.st_size//2-32768),max(0,st.st_size-65536)]:
                f.seek(pos);h.update(f.read(65536))
        row['sampled_source_fingerprint_sha256']=h.hexdigest();rows.append(row)
    groups=collections.defaultdict(set)
    for r in rows:groups[(r['driver'],r['session'])].add(r['view'])
    assert rows and all(v=={'dashboard','rearview','right_side_window'} for v in groups.values())
    cfg=dict(raw_root=str(raw),output_root=str(out),role=role,expected_videos=len(rows),expected_recordings=len(groups),
        labels_root=r'F:\labels&instructions',seed=42,held_out_drivers_per_training_part=2,
        source_videos_read_only=True,do_not_train_on_A2=True)
    (out/'package_config.json').write_text(json.dumps(cfg,indent=2))
    (out/'source_videos.json').write_text(json.dumps(rows,indent=2))
    (out/'source_inventory_original.json').write_text(json.dumps(rows,indent=2))
    scripts=['extract_motion.py','fit_alignment.py','test_alignment.py']
    if role=='A1_training':scripts+=['make_clip_plan.py','export_clips.py','validate_exports.py','verify_training.py','training_dataset.py','create_previews.py']
    for name in scripts:
        text=(BASE/name).read_text()
        if name=='extract_motion.py':
            text=text.replace('count=0;last_report=0','count=0;last_report=0;first_pts=None;last_pts=None;corrupt_frames=0')
            text=text.replace('a=frame.reformat(width=128',"first_pts=t if first_pts is None else first_pts;last_pts=t;corrupt_frames+=int(frame.is_corrupt)\n            a=frame.reformat(width=128")
            text=text.replace("feature_names=['global_motion_without_clock'", "first_pts_s=first_pts,last_pts_s=last_pts,decoded_duration_s=(last_pts+1/30 if last_pts is not None else 0),corrupt_frames=corrupt_frames,\n              feature_names=['global_motion_without_clock'")
        if name=='validate_exports.py':
            text=text.replace("if extraction['done']!=60", "if extraction['done']!=len(json.loads((OUT/'source_videos.json').read_text()))")
            text=text.replace("if len(mappings)!=20", "if len(mappings)!=len({(r['driver'],r['session']) for r in json.loads((OUT/'source_videos.json').read_text())})")
            text=text.replace("source_recordings=20", "source_recordings=len(mappings)")
        if name=='verify_training.py':
            text=text.replace('assert len(source_features)==60',"assert len(source_features)==len(json.loads((OUT/'source_videos.json').read_text()))")
        (out/name).write_text(text)
    if role=='A1_training':shutil.copy2(HERE/'audit_labels_generic.py',out/'audit_labels.py')
    print(json.dumps(dict(root=str(raw),output=str(out),videos=len(rows),drivers=len({r['driver'] for r in rows}),recordings=len(groups),GB=sum(r['source_size_bytes'] for r in rows)/1e9),indent=2),flush=True)

if __name__=='__main__':
    for part in ['A1_2','A1_3','A1_4']:initialize(Path('F:/')/part,Path('F:/')/(part+'_preprocessed'),'A1_training')
    initialize(Path(r'F:\A2'),Path(r'F:\A2_preprocessed'),'A2_unlabelled_evaluation')
