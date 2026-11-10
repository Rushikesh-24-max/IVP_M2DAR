"""Meaningful checks for alignment sign, changing offsets and uninformative data."""
from pathlib import Path
import sys,json
OUT=Path(__file__).resolve().parent;sys.path.insert(0,r'F:\A1_analysis\tools')
import numpy as np
from fit_alignment import estimate_pair

def signal():
    rng=np.random.default_rng(2024);t=np.arange(0,120,1/30);x=np.zeros_like(t)
    for center in np.arange(2,118,2.7):
        x+=float(rng.uniform(0.5,2))*np.exp(-0.5*((t-center)/float(rng.uniform(.12,.4)))**2)
    return t,x

def run():
    t,x=signal();a={'t':t,'f':np.stack([x,x*1.2,x*1.7],axis=1)};results=[]
    for lag in [-9,9]:
        y=np.interp(t-lag/30,t,x,left=0,right=0);b={'t':t,'f':np.stack([y,y*.8,y*2],axis=1)}
        windows,segments,_=estimate_pair(a,b,120)
        accepted=[w['offset_frames'] for w in windows if w['accepted']]
        assert accepted and abs(np.median(accepted)-lag)<=1,(lag,accepted)
        assert segments and all(abs(s['offset_frames']-lag)<=1 for s in segments)
        results.append(dict(check='known_offset_sign',offset_frames=lag,passed=True))
    y=np.where(t<60,np.interp(t+.3,t,x),np.interp(t-.6,t,x));b={'t':t,'f':np.stack([y,y*.8,y*2],axis=1)}
    _,segments,_=estimate_pair(a,b,120);lags={s['offset_frames'] for s in segments}
    assert any(abs(z+9)<=1 for z in lags) and any(abs(z-18)<=1 for z in lags),lags
    results.append(dict(check='piecewise_change_preserved',offset_frames=sorted(lags),passed=True))
    flat={'t':t,'f':np.zeros((len(t),3))};_,segments,_=estimate_pair(flat,flat,120)
    assert not segments
    results.append(dict(check='static_signals_not_forced_to_zero_offset',passed=True))
    (OUT/'alignment_test_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':run()
