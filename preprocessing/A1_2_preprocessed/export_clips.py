"""Export synchronized full-field-of-view action clips; original sources stay read-only."""
from pathlib import Path
from fractions import Fraction
import sys,json,time,argparse,concurrent.futures,hashlib
OUT=Path(__file__).resolve().parent;sys.path.insert(0,r'F:\A1_analysis\tools')
import av
FPS=30

def export_view(job,view,device=None):
    spec=job['views'][view];path=OUT/spec['relative_output_path'];meta_path=path.with_suffix('.json')
    signature=hashlib.sha256(json.dumps(dict(spec=spec,frames=job['frame_count'],width=960,height=540,fps=FPS,crf=20),sort_keys=True).encode()).hexdigest()
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta.get('signature')==signature:return dict(meta,cached=True)
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.stem+'.partial.mp4')
    started=time.time();maximum_error=0;sampled=0
    options={'hwaccel':av.codec.hwaccel.HWAccel(device,allow_software_fallback=False)} if device else {}
    with av.open(spec['source_path'],**options) as source:
        stream=source.streams.video[0];stream.codec_context.thread_count=2
        start=spec['source_start_s'];source.seek(max(0,int((start-0.1)/stream.time_base)),stream=stream,backward=True,any_frame=False)
        iterator=source.decode(stream);prev=None;current=next(iterator)
        with av.open(str(temp),'w',format='mp4',options={'movflags':'+faststart'}) as target:
            encoded=target.add_stream('libx264',rate=FPS);encoded.width=960;encoded.height=540;encoded.pix_fmt='yuv420p'
            encoded.codec_context.thread_count=2;encoded.gop_size=30;encoded.options={'crf':'20','preset':'veryfast'}
            for i in range(job['frame_count']):
                wanted=start+i/FPS
                while current.time is None or float(current.time)<wanted-1e-7:
                    prev=current
                    try:current=next(iterator)
                    except StopIteration:raise RuntimeError(f'Source ended before requested sample {wanted:.6f}')
                chosen=prev if prev is not None and prev.time is not None and abs(float(prev.time)-wanted)<abs(float(current.time)-wanted) else current
                error=abs(float(chosen.time)-wanted);maximum_error=max(maximum_error,error)
                if error>1/FPS+0.002:raise RuntimeError(f'PTS gap: wanted {wanted:.6f}, got {chosen.time}')
                frame=chosen.reformat(width=960,height=540,format='yuv420p');frame.pts=i;frame.time_base=Fraction(1,FPS)
                for packet in encoded.encode(frame):target.mux(packet)
                sampled+=1
            for packet in encoded.encode():target.mux(packet)
    temp.replace(path)
    meta=dict(group_id=job['group_id'],driver=job['driver'],session=job['session'],view=view,label=job['label'],split=job['split'],path=str(path),relative_path=spec['relative_output_path'],frame_count=sampled,fps=FPS,width=960,height=540,
        reference_start_s=job['reference_start_s'],source_start_s=start,offset_s=spec['offset_s'],max_source_sample_time_error_s=maximum_error,source_pts_sampling='nearest presentation timestamp',
        signature=signature,size_bytes=path.stat().st_size,wall_s=time.time()-started,status='exported_pending_validation')
    meta_path.write_text(json.dumps(meta,indent=2));return meta

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit-groups',type=int);p.add_argument('--device',default=None);args=p.parse_args()
    jobs=json.loads((OUT/'clip_plan.json').read_text())['groups']
    if args.limit_groups:jobs=jobs[:args.limit_groups]
    tasks=[(job,view) for job in jobs for view in ['dashboard','rearview','right_side_window']]
    results=[];begin=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(export_view,job,view,args.device):(job,view) for job,view in tasks}
        for future in concurrent.futures.as_completed(pending):
            job,view=pending[future]
            try:r=future.result()
            except Exception as e:r=dict(group_id=job['group_id'],view=view,error=repr(e));print('ERROR',r,flush=True)
            results.append(r)
            if len(results)%12==0 or len(results)==len(tasks):print('EXPORTED',len(results),'/',len(tasks),'elapsed',round(time.time()-begin,1),flush=True)
            (OUT/'export_progress.json').write_text(json.dumps(dict(done=len(results),total=len(tasks),elapsed_s=time.time()-begin,results=results),indent=2))
    print('Export complete; errors',sum('error' in r for r in results),flush=True)
