"""Validate only annotations belonging to the user's A1 video subset."""
from pathlib import Path
import csv,json,re,hashlib,collections,sys
OUT=Path(__file__).resolve().parent
RAW=Path(r'F:\A1_extracted')
LABELS=Path(r'F:\labels&instructions')
VIEWS=['dashboard','rearview','right_side_window']

def identify(name):
    s=str(name).strip().lower().replace('noaudio_','').replace('user_id_','')
    m=re.search(r'(\d+)_(\d+)(?:\.[^.]+)?$',s)
    if not m: raise ValueError('Unrecognized video filename: '+str(name))
    view='dashboard' if s.startswith('dashboard') else 'rearview' if s.startswith('rear') else 'right_side_window' if s.startswith('right') else None
    if view is None: raise ValueError('Unrecognized camera: '+str(name))
    return m[1],m[2],view

def seconds(s):
    parts=[float(x) for x in s.strip().split(':')]
    if len(parts)!=3 or min(parts)<0 or parts[1]>=60 or parts[2]>=60: raise ValueError('Invalid time: '+s)
    return parts[0]*3600+parts[1]*60+parts[2]

def run():
    OUT.mkdir(exist_ok=True)
    videos={identify(p.name):str(p) for p in sorted(RAW.rglob('*.MP4'))}
    previous=json.loads(Path(r'F:\A1_analysis\decoded_audit.json').read_text())
    durations={(r['driver'],r['session'],r['view']):r['playback_duration_s'] for r in previous}
    records=[];issues=[];hashes={};drivers=sorted({k[0] for k in videos})
    for driver in drivers:
        p=LABELS/'A1'/f'user_id_{driver}.csv'
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        prevname=None
        with p.open(encoding='utf-8-sig',newline='') as f:
            for number,row in enumerate(csv.DictReader(f),2):
                name=row.get('Filename','').strip() or prevname
                if not name:
                    issues.append(dict(driver=driver,row=number,issue='missing_filename'));continue
                prevname=name
                key=identify(name)
                if key not in videos:
                    issues.append(dict(driver=driver,row=number,filename=name,issue='no_matching_video'));continue
                rawclass=row.get('Label (Primary)',row.get('Label','')).strip()
                m=re.fullmatch(r'Class\s*(\d+)',rawclass,re.I)
                if not m:
                    issues.append(dict(driver=driver,row=number,filename=name,label=rawclass,issue='unlabeled_or_invalid_class'));continue
                label=int(m[1]);a=seconds(row['Start Time']);b=seconds(row['End Time'])
                reasons=[]
                if not 0<=label<16:reasons.append('invalid_class_id')
                if a>=b:reasons.append('nonpositive_interval')
                if a<0 or a>=durations[key] or b>durations[key]+1e-5:reasons.append('outside_video_duration')
                record=dict(driver=key[0],session=key[1],view=key[2],label=label,start_s=a,end_s=b,
                    source_path=videos[key],source_annotation_file=str(p),source_row=number,
                    original_filename=name,appearance=row.get('Appearance Block',''),
                    playback_duration_s=durations[key],status='excluded' if reasons else 'valid',issues=reasons)
                records.append(record)
                for reason in reasons:issues.append(dict(driver=driver,row=number,filename=name,label=label,start_s=a,end_s=b,issue=reason))
    # Quarantine a row spanning multiple separately annotated actions. Do not
    # guess a replacement time or let a long erroneous label overwrite them.
    for key in videos:
        rr=[r for r in records if (r['driver'],r['session'],r['view'])==key]
        for x in rr:
            covered=[y for y in rr if y is not x and x['start_s']<y['start_s']<x['end_s'] and y['label']!=x['label']]
            if len(covered)>=2 and x['status']=='valid':
                x['issues'].append('spans_multiple_labeled_actions');x['status']='excluded'
                issues.append(dict(driver=key[0],session=key[1],view=key[2],row=x['source_row'],label=x['label'],issue='spans_multiple_labeled_actions',covered_rows=[y['source_row'] for y in covered]))
    for key in videos:
        rr=sorted((r for r in records if (r['driver'],r['session'],r['view'])==key and r['status']=='valid'),key=lambda r:r['start_s'])
        for i,x in enumerate(rr):
            for y in rr[i+1:]:
                if y['start_s']>=x['end_s']:break
                issues.append(dict(driver=key[0],session=key[1],view=key[2],issue='overlapping_annotations',rows=[x['source_row'],y['source_row']]))
    summary=dict(video_files=len(videos),drivers=drivers,annotation_files=len(hashes),annotation_rows=len(records),
        valid_rows=sum(r['status']=='valid' for r in records),excluded_rows=sum(r['status']!='valid' for r in records),
        issues=dict(collections.Counter(r['issue'] for r in issues)),
        class_rows=dict(collections.Counter(r['label'] for r in records if r['status']=='valid')),
        source_annotation_sha256=hashes,
        scope='Only ten drivers whose videos exist in A1_extracted; other CSVs were not analyzed.')
    (OUT/'annotations_normalized.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (OUT/'annotation_issues.json').write_text(json.dumps(issues,indent=2),encoding='utf-8')
    (OUT/'annotation_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'source_videos.json').write_text(json.dumps([dict(driver=k[0],session=k[1],view=k[2],path=v,duration_s=durations[k]) for k,v in videos.items()],indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Issue examples:',json.dumps(issues[:12],indent=2))

if __name__=='__main__':run()
