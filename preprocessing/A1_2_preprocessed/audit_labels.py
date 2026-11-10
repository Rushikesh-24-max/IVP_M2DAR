"""Validate supplied labels without guessing missing classes or correcting times."""
from pathlib import Path
import csv,json,re,hashlib,collections
OUT=Path(__file__).resolve().parent
VIEWS=['dashboard','rearview','right_side_window']

def identify(name):
    s=str(name).strip().lower().replace('noaudio_','').replace('user_id_','')
    m=re.search(r'(\d+)_(\d+)(?:\.[^.]+)?$',s)
    if not m:raise ValueError('Unrecognized video filename: '+str(name))
    v='dashboard' if s.startswith('dashboard') else 'rearview' if s.startswith('rear') else 'right_side_window' if s.startswith('right') else None
    if v is None:raise ValueError('Unrecognized camera: '+str(name))
    return m[1],m[2],v

def seconds(value):
    p=[float(x) for x in value.strip().split(':')]
    if len(p)!=3 or min(p)<0 or p[1]>=60 or p[2]>=60:raise ValueError('Invalid time: '+value)
    return p[0]*3600+p[1]*60+p[2]

def run():
    cfg=json.loads((OUT/'package_config.json').read_text());sources=json.loads((OUT/'source_videos.json').read_text())
    videos={(r['driver'],r['session'],r['view']):r for r in sources}
    labels=Path(cfg['labels_root'])/'A1';records=[];issues=[];hashes={};corrections=[]
    def exclude(row,reason,**extra):
        if reason not in row['issues']:row['issues'].append(reason)
        row['status']='excluded'
        issues.append(dict(driver=row['driver'],session=row['session'],view=row['view'],row=row['source_row'],label=row['label'],start_s=row['start_s'],end_s=row['end_s'],issue=reason,**extra))
    for driver in sorted({r['driver'] for r in sources}):
        p=labels/f'user_id_{driver}.csv';hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();previous=None
        with p.open(encoding='utf-8-sig',newline='') as f:
            for number,row in enumerate(csv.DictReader(f),2):
                row={str(k).strip():str(v or '').strip() for k,v in row.items() if k is not None}
                name=row.get('Filename') or previous;previous=name
                try:
                    key=identify(name);filename_key=key
                    declared=re.sub(r'[^a-z]','',row.get('Camera View','').lower())
                    declared_view='dashboard' if declared.startswith('dash') else 'rearview' if declared.startswith('rear') else 'right_side_window' if declared.startswith('right') else None
                    camera_conflict=bool(declared_view and declared_view!=key[2])
                    # This supplied CSV repeats the dashboard filename in all three
                    # explicit camera blocks. Resolve only this inspected defect.
                    if camera_conflict and driver=='47158' and key[0]=='47158' and key[2]=='dashboard':
                        key=(key[0],key[1],declared_view);camera_conflict=False
                        corrections.append(dict(driver=driver,source_annotation_file=str(p),source_row=number,
                            original_filename=name,declared_camera_view=row['Camera View'],resolved_key=list(key),
                            reason='47158 repeated dashboard filename; explicit Camera View column identifies the actual camera',
                            action_label_and_times_unchanged=True))
                    source=videos.get(key)
                    if source is None:raise ValueError('no_matching_video')
                    m=re.fullmatch(r'Class\s*(\d+)',row.get('Label (Primary)',row.get('Label','')),re.I)
                    if not m:raise ValueError('unlabeled_or_invalid_class')
                    label=int(m[1]);a=seconds(row['Start Time']);b=seconds(row['End Time'])
                except Exception as e:
                    issues.append(dict(driver=driver,row=number,filename=name,issue='unparseable_or_unmatched_annotation',detail=str(e)));continue
                r=dict(driver=key[0],session=key[1],view=key[2],label=label,start_s=a,end_s=b,source_path=source['path'],
                    source_annotation_file=str(p),source_row=number,original_filename=name,appearance=row.get('Appearance Block',''),
                    declared_camera_view=row.get('Camera View'),filename_camera_view=filename_key[2],camera_mapping_corrected=key!=filename_key,
                    playback_duration_s=source['duration_s'],status='valid',issues=[])
                records.append(r)
                if not 0<=label<16:exclude(r,'invalid_class_id')
                if a>=b:exclude(r,'nonpositive_interval')
                if a<0 or a>=source['duration_s'] or b>source['duration_s']+1e-5:exclude(r,'outside_video_duration')
                if driver!=key[0]:exclude(r,'annotation_driver_mismatch')
                if camera_conflict:exclude(r,'unresolved_camera_column_filename_conflict')
    for key in videos:
        rr=[r for r in records if (r['driver'],r['session'],r['view'])==key]
        seen=set()
        for x in rr:
            identity=(x['start_s'],x['end_s'],x['label'])
            if identity in seen:exclude(x,'duplicate_annotation')
            seen.add(identity)
            covered=[y for y in rr if y is not x and x['start_s']<y['start_s']<x['end_s'] and y['label']!=x['label']]
            if len(covered)>=2 and x['status']=='valid':exclude(x,'spans_multiple_labeled_actions',covered_rows=[y['source_row'] for y in covered])
        rr=sorted((r for r in rr if r['status']=='valid'),key=lambda r:r['start_s'])
        ambiguous=[]
        for i,x in enumerate(rr):
            for y in rr[i+1:]:
                if y['start_s']>=x['end_s']:break
                if y['label']!=x['label']:ambiguous.extend([(x,y['source_row']),(y,x['source_row'])])
        for x,other in ambiguous:exclude(x,'conflicting_annotation_overlap',other_row=other)
    summary=dict(video_files=len(videos),drivers=sorted({r['driver'] for r in sources}),annotation_files=len(hashes),annotation_rows=len(records),
        valid_rows=sum(r['status']=='valid' for r in records),excluded_rows=sum(r['status']!='valid' for r in records),
        issues=dict(collections.Counter(x['issue'] for x in issues)),class_rows=dict(collections.Counter(r['label'] for r in records if r['status']=='valid')),
        source_annotation_sha256=hashes,camera_mapping_corrections=len(corrections),scope=cfg['raw_root'],policy='Supplied labels only; inspected 47158 camera-prefix correction logged; no A2 labels, inferred class-0 gaps or guessed action/time corrections.')
    for name,obj in [('annotations_normalized.json',records),('annotation_issues.json',issues),('annotation_corrections.json',corrections),('annotation_audit.json',summary)]:
        (OUT/name).write_text(json.dumps(obj,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
