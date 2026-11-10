"""Restore segment-local localization predictions to original A2 dashboard time."""
from pathlib import Path
import json,argparse,collections,math

def restore(rows,groups):
    by={g['group_id']:g for g in groups};candidates=[]
    for number,line in enumerate(rows,1):
        if not line.strip():continue
        gid,label,a,b=line.split();g=by[gid];label=int(label);a=float(a);b=float(b)
        if not 0<=label<16 or not math.isfinite(a+b) or a>b:raise ValueError('Invalid prediction row '+str(number))
        # Baseline (0,0) is its no-candidate sentinel, not an activity at time 0.
        if a==b:continue
        duration=g['frame_count']/30;a=max(0,a);b=min(duration,b)
        if a>=b:continue
        candidates.append(dict(video_id=g['video_id'],class_id=label,start_s=g['reference_start_s']+a,
            end_s=g['reference_start_s']+b,source_segments=[gid]))
    buckets=collections.defaultdict(list)
    for r in candidates:buckets[(r['video_id'],r['class_id'])].append(r)
    result=[]
    for key,rr in sorted(buckets.items()):
        current=None
        for r in sorted(rr,key=lambda r:(r['start_s'],r['end_s'])):
            # Only overlapping intervals merge. A gap with unknown footage is preserved.
            if current is not None and r['start_s']<=current['end_s']:
                current['end_s']=max(current['end_s'],r['end_s']);current['source_segments']=sorted(set(current['source_segments']+r['source_segments']))
            else:
                if current:result.append(current)
                current=dict(r)
        if current:result.append(current)
    return sorted(result,key=lambda r:(r['video_id'],r['start_s'],r['class_id']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--root',default=str(Path(__file__).resolve().parent));a=p.parse_args()
    groups=[json.loads(x) for x in (Path(a.root)/'inference_groups.jsonl').read_text().splitlines() if x.strip()]
    result=restore(Path(a.input).read_text().splitlines(),groups)
    Path(a.output).write_text(''.join(json.dumps(x)+'\n' for x in result));print('Restored candidates:',len(result))
