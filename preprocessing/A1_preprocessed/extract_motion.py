"""Read original A1 presentation times and dense visual motion; never rewrite sources."""
from pathlib import Path
import sys,json,time,argparse,concurrent.futures
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(r'F:\A1_analysis\tools')))
import av,numpy as np

def key(r):return f"{r['driver']}_{r['session']}_{r['view']}"
def process(r,limit=None,device=None):
    begin=time.time();k=key(r);target=OUT/'dense_motion'/f'{k}.npz'
    stat=Path(r['path']).stat()
    if limit is None and target.exists():
        old=np.load(target,allow_pickle=False)
        meta=json.loads(str(old['metadata']))
        if meta.get('source_size_bytes')==stat.st_size and meta.get('source_mtime_ns')==stat.st_mtime_ns:
            return dict(key=k,cached=True,frames=len(old['t']))
    times=[];features=[];bad=[];prev=None;pt=None;count=0;last_report=0
    rois={'dashboard':(12,66,38,108),'rearview':(16,66,55,120),'right_side_window':(15,66,25,96)}
    y0,y1,x0,x1=rois[r['view']]
    options={'hwaccel':av.codec.hwaccel.HWAccel(device,allow_software_fallback=False)} if device else {}
    with av.open(r['path'],**options) as c:
        stream=c.streams.video[0];stream.codec_context.thread_count=2
        for frame in c.decode(stream):
            t=float(frame.time) if frame.time is not None else None
            if t is None or t<0:continue
            if limit is not None and t>=limit:break
            a=frame.reformat(width=128,height=72,format='gray').to_ndarray().astype(np.float32)
            count+=1
            if pt is not None:
                dt=t-pt
                if dt<=0:
                    bad.append(dict(type='nonincreasing_pts',previous_s=pt,time_s=t));continue
                if abs(dt-1/30)>0.002:
                    bad.append(dict(type='irregular_pts_interval',previous_s=pt,time_s=t,delta_s=dt))
                d=np.maximum(np.abs(a-prev)-0.5,0)
                body=d[:60,:]
                grid=body.reshape(4,15,8,16).mean(axis=(1,3)).ravel()
                # Three motion signals; time selection always uses decoded PTS.
                features.append([float(body.mean()),float(d[y0:y1,x0:x1].mean()),float(np.sort(grid)[-5:].mean())])
                times.append(t)
            prev=a;pt=t
            if t-last_report>=90:
                print(f'PROGRESS {k} {t:.1f}/{r["duration_s"]:.1f}s',flush=True);last_report=t
    meta=dict(key=k,source_path=r['path'],source_size_bytes=stat.st_size,source_mtime_ns=stat.st_mtime_ns,
              decoded_frames=count,feature_frames=len(times),irregular_pts=bad,wall_s=time.time()-begin,
              feature_names=['global_motion_without_clock','driver_region_motion','top5_grid_motion'],
              pixels=[128,72],complete=limit is None,decoder_device=device or 'software')
    if limit is None:
        target.parent.mkdir(exist_ok=True)
        temp=target.with_suffix('.partial.npz')
        np.savez_compressed(temp,t=np.asarray(times,dtype=np.float64),f=np.asarray(features,dtype=np.float32),metadata=json.dumps(meta))
        temp.replace(target)
    return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--benchmark',action='store_true');parser.add_argument('--device',default=None);args=parser.parse_args()
    rows=json.loads((OUT/'source_videos.json').read_text())
    if args.benchmark:
        print('Encoders:',{x:x in av.codecs_available for x in ['h264_qsv','h264_nvenc','libx264']},flush=True)
        annotations=json.loads((OUT/'annotations_normalized.json').read_text())
        print('Overlap example:',[(x['label'],x['start_s'],x['end_s']) for x in annotations if x['driver']=='25432' and x['session']=='5' and x['view']=='dashboard'],flush=True)
        print(json.dumps(process(rows[0],10,args.device),indent=2),flush=True)
    else:
        begin=time.time();results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            fs={pool.submit(process,r,None,args.device):r for r in rows}
            for fut in concurrent.futures.as_completed(fs):
                try:
                    result=fut.result();results.append(result)
                    print('DONE',len(results),'/',len(rows),result['key'],'frames',result.get('decoded_frames',result.get('frames')),'elapsed',round(time.time()-begin,1),flush=True)
                except Exception as e:
                    result=dict(key=key(fs[fut]),error=repr(e));results.append(result);print('ERROR',result,flush=True)
                (OUT/'dense_motion_progress.json').write_text(json.dumps(dict(done=len(results),total=len(rows),elapsed_s=time.time()-begin,results=results),indent=2))
        print('COMPLETE',round(time.time()-begin,1),'seconds',flush=True)
