"""Estimate conservative piecewise camera mappings from dense PTS motion evidence.

Source view time = dashboard reference time + offset. Export only stable supported
intervals; uncertain transitions and long unsupported gaps are excluded.
"""
from pathlib import Path
import sys,json,csv,argparse,collections
OUT=Path(__file__).resolve().parent;sys.path.insert(0,r'F:\A1_analysis\tools')
import numpy as np
FPS=30;HALF=7.5;SEARCH=45;STEP=4.0

def smooth(x):return np.convolve(x,np.ones(3)/3,mode='same')
def correlations(a,b):
    n=len(a);a=a-a.mean();an=np.sqrt(np.dot(a,a))
    if an<1e-6:return np.zeros(len(b)-n+1)
    cs=np.r_[0,np.cumsum(b)];cs2=np.r_[0,np.cumsum(b*b)]
    sums=cs[n:]-cs[:-n];sums2=cs2[n:]-cs2[:-n]
    den=np.sqrt(np.maximum(sums2-sums*sums/n,1e-10))*an
    return np.correlate(b,a,mode='valid')/den

def subtract(intervals,holes):
    result=list(intervals)
    for a,b in holes:
        nextresult=[]
        for x,y in result:
            if b<=x or a>=y:nextresult.append((x,y));continue
            if a>x:nextresult.append((x,min(a,y)))
            if b<y:nextresult.append((max(b,x),y))
        result=nextresult
    return [(max(0,a),b) for a,b in result if b-a>=64/30]

def estimate_pair(ref,other,duration):
    grid=np.arange(0,duration,1/FPS)
    A=np.stack([smooth(np.interp(grid,ref['t'],ref['f'][:,i])) for i in range(3)],axis=1)
    B=np.stack([smooth(np.interp(grid,other['t'],other['f'][:,i])) for i in range(3)],axis=1)
    windows=[]
    for center in np.arange(10,duration-9.99,STEP):
        i=int(round(center*FPS));lo=i-int(HALF*FPS);hi=i+int(HALF*FPS)
        if lo-SEARCH<0 or hi+SEARCH>len(grid):continue
        curves=[correlations(A[lo:hi,j],B[lo-SEARCH:hi+SEARCH,j]) for j in range(3)]
        score=(curves[0]+curves[2])/2
        peak=int(np.argmax(score));lag=peak-SEARCH;best=float(score[peak])
        away=np.abs(np.arange(len(score))-peak)>=9
        margin=best-float(score[away].max())
        agreement=abs(int(np.argmax(curves[0]))-int(np.argmax(curves[2])))
        x=A[lo:hi,0];y=B[lo-SEARCH+peak:hi-SEARCH+peak,0]
        informative=min(float(x.std()),float(y.std()))>=0.015
        # Effective support prevents one compression spike defining the entire lag.
        effective=min(float(x.sum()**2/max(np.dot(x,x),1e-8)),float(y.sum()**2/max(np.dot(y,y),1e-8)))
        good=(best>=0.65 and min(curves[0][peak],curves[2][peak])>=0.55 and margin>=0.06
              and agreement<=3 and informative and effective>=12 and abs(lag)<SEARCH-1)
        windows.append(dict(center_s=round(float(center),3),offset_frames=lag,offset_s=lag/FPS,
            correlation=best,peak_margin=margin,signal_agreement_frames=agreement,
            effective_support_frames=effective,accepted=bool(good)))
    good=[w for w in windows if w['accepted']]
    if len(good)<3:return windows,[],[dict(reason='insufficient_informative_motion_windows',start_s=0,end_s=duration)]
    # Penalized robust segmentation: isolated projection/occlusion differences
    # must not create a camera edit. A persistent change can justify a new block.
    n=len(good);values=np.asarray([p['offset_frames'] for p in good]);penalty=36.0
    dp=np.full(n+1,np.inf);back=np.full(n+1,-1,dtype=int);dp[0]=-penalty
    for j in range(1,n+1):
        for i in range(j):
            if not np.isfinite(dp[i]):continue
            if j-i<5 and not (i==0 and j==n):continue
            if good[j-1]['center_s']-good[i]['center_s']<16 and not (i==0 and j==n):continue
            segment=values[i:j];median=float(np.median(segment))
            cost=float(np.minimum(np.abs(segment-median),6).sum())
            total=dp[i]+cost+penalty
            if total<dp[j]:dp[j]=total;back[j]=i
    clusters=[];j=n
    while j>0 and back[j]>=0:
        i=int(back[j]);clusters.append(good[i:j]);j=i
    clusters=list(reversed(clusters)) if j==0 else [good]
    segments=[];quarantine=[]
    for ci,c in enumerate(clusters):
        offset=int(round(float(np.median([p['offset_frames'] for p in c]))))
        consistent=[p for p in c if abs(p['offset_frames']-offset)<=2]
        if len(consistent)<3:continue
        start=max(0,c[0]['center_s']-25);end=min(duration,c[-1]['center_s']+25)
        if ci>0:
            change_end=c[0]['center_s']+HALF
            change_start=clusters[ci-1][-1]['center_s']-HALF
            start=max(start,change_end)
            quarantine.append(dict(reason='uncertain_offset_transition',start_s=max(0,change_start),end_s=min(duration,change_end)))
        if ci+1<len(clusters):end=min(end,c[-1]['center_s']-HALF)
        holes=[]
        for x,y in zip(consistent,consistent[1:]):
            if y['center_s']-x['center_s']>60:
                holes.append((x['center_s']+10,y['center_s']-10))
                quarantine.append(dict(reason='long_gap_without_informative_motion',start_s=holes[-1][0],end_s=holes[-1][1]))
        # Reject a sustained inconsistent run, not isolated noisy correlation
        # peaks. Camera viewpoint can change when a hand/head becomes visible.
        run=[]
        for p in c+[None]:
            if p is not None and abs(p['offset_frames']-offset)>2:
                run.append(p);continue
            if len(run)>=3 and run[-1]['center_s']-run[0]['center_s']>=8:
                holes.append((run[0]['center_s']-HALF,run[-1]['center_s']+HALF))
            run=[]
        for a,b in subtract([(start,end)],holes):
            anchors=[p for p in consistent if a-HALF<=p['center_s']<=b+HALF]
            if len(anchors)<2:continue
            segments.append(dict(reference_start_s=a,reference_end_s=b,offset_frames=offset,offset_s=offset/FPS,
                accepted_anchor_count=len(anchors),max_anchor_residual_frames=max(abs(p['offset_frames']-offset) for p in anchors),
                robust_fit_outliers=sum(abs(p['offset_frames']-offset)>2 for p in c),
                anchor_centers_s=[p['center_s'] for p in anchors],status='estimated_pending_output_checks'))
    if not segments:quarantine.append(dict(reason='no_stable_supported_mapping',start_s=0,end_s=duration))
    return windows,segments,quarantine

def run(available=False):
    videos=json.loads((OUT/'source_videos.json').read_text());groups=collections.defaultdict(dict)
    for r in videos:groups[(r['driver'],r['session'])][r['view']]=r
    mappings=[];allwindows=[];exclusions=[]
    for (driver,session),views in sorted(groups.items()):
        files={v:OUT/'dense_motion'/f'{driver}_{session}_{v}.npz' for v in views}
        if not all(p.exists() for p in files.values()):
            if available:continue
            raise RuntimeError('Dense extraction incomplete: '+driver+'/'+session)
        loaded={v:np.load(p,allow_pickle=False) for v,p in files.items()}
        duration=min(r['duration_s'] for r in views.values())
        mapping=dict(driver=driver,session=session,reference_view='dashboard',views={})
        mapping['views']['dashboard']=dict(source_path=views['dashboard']['path'],segments=[dict(reference_start_s=0,reference_end_s=duration,offset_frames=0,offset_s=0,status='reference')])
        for view in ['rearview','right_side_window']:
            windows,segments,bad=estimate_pair(loaded['dashboard'],loaded[view],duration)
            # Irregular presentation-time intervals are not interpolated into clips.
            irregular=json.loads(str(loaded[view]['metadata'])).get('irregular_pts',[])
            ref_irregular=json.loads(str(loaded['dashboard']['metadata'])).get('irregular_pts',[])
            clean=[]
            for seg in segments:
                holes=[(p['time_s']-seg['offset_s']-1,p['time_s']-seg['offset_s']+1) for p in irregular]
                holes.extend((p['time_s']-1,p['time_s']+1) for p in ref_irregular)
                for a,b in subtract([(seg['reference_start_s'],seg['reference_end_s'])],holes):
                    clean.append(dict(seg,reference_start_s=a,reference_end_s=b))
            mapping['views'][view]=dict(source_path=views[view]['path'],segments=clean)
            for w in windows:allwindows.append(dict(driver=driver,session=session,view=view,**w))
            for w in bad:exclusions.append(dict(driver=driver,session=session,view=view,**w))
        mappings.append(mapping)
        print(driver,session,{v:[(round(s['reference_start_s'],1),round(s['reference_end_s'],1),s['offset_frames']) for s in info['segments']] for v,info in mapping['views'].items()},flush=True)
    (OUT/'alignment_estimates.json').write_text(json.dumps(dict(schema_version=1,fps=FPS,sign='source_time = reference_time + offset',method='dense PTS motion, fixed global/top-grid signal, conservative piecewise constants',groups=mappings),indent=2))
    (OUT/'alignment_windows.json').write_text(json.dumps(allwindows,indent=2))
    (OUT/'alignment_quarantine.json').write_text(json.dumps(exclusions,indent=2))
    if allwindows:
        with (OUT/'alignment_windows.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(allwindows[0]));w.writeheader();w.writerows(allwindows)
    print('Groups fitted',len(mappings),'accepted windows',sum(x['accepted'] for x in allwindows),'/',len(allwindows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');a=p.parse_args();run(a.available)
