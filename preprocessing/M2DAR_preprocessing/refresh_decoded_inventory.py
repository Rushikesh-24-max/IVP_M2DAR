"""Reconcile source metadata with the complete decoded presentation timeline."""
from pathlib import Path
import json,sys
sys.path.insert(0,r'F:\A1_analysis\tools')
import numpy as np

def run(root):
    rows=json.loads((root/'source_videos.json').read_text());report=[]
    progress=json.loads((root/'dense_motion_progress.json').read_text())
    if progress['done']!=len(rows) or any('error' in x for x in progress['results']):raise RuntimeError('Source decode pass incomplete or failed: '+str(root))
    for r in rows:
        p=root/'dense_motion'/f"{r['driver']}_{r['session']}_{r['view']}.npz"
        with np.load(p,allow_pickle=False) as z:meta=json.loads(str(z['metadata']));t=z['t']
        if not meta['complete'] or meta.get('corrupt_frames',0):raise RuntimeError('Incomplete/corrupt source: '+r['path'])
        duration=meta.get('decoded_duration_s',float(t[-1])+1/30)
        r.update(duration_s=duration,decoded_frames=meta['decoded_frames'],first_pts_s=meta.get('first_pts_s',0),
            last_pts_s=meta.get('last_pts_s',float(t[-1])),irregular_pts_count=len(meta['irregular_pts']),corrupt_frames=meta.get('corrupt_frames',0))
        report.append(dict(path=r['path'],header_duration_s=r['header_duration_s'],decoded_duration_s=duration,
            frames=meta['decoded_frames'],irregular_pts_count=len(meta['irregular_pts']),corrupt_frames=meta.get('corrupt_frames',0)))
    (root/'source_videos.json').write_text(json.dumps(rows,indent=2))
    (root/'decoded_source_audit.json').write_text(json.dumps(report,indent=2))
    print('Full decode reconciled:',root.name,len(rows),'files; irregular PTS files',sum(x['irregular_pts_count']>0 for x in report),flush=True)

if __name__=='__main__':run(Path(sys.argv[1]))
