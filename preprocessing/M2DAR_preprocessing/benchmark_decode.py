from pathlib import Path
import sys,time,json
sys.path.insert(0,r'F:\A1_analysis\tools')
import av,numpy as np

path=next(Path(r'F:\A1_2').rglob('*.MP4'))
for device in ['d3d11va',None]:
    for mode in ['reformat','luma']:
        t=time.time();count=0
        opts={'hwaccel':av.codec.hwaccel.HWAccel(device,allow_software_fallback=False)} if device else {}
        with av.open(str(path),**opts) as c:
            s=c.streams.video[0];s.codec_context.thread_count=2
            for f in c.decode(s):
                if float(f.time)>=20:break
                if mode=='reformat':a=f.reformat(width=128,height=72,format='gray').to_ndarray()
                else:
                    p=f.planes[0]
                    a=np.frombuffer(p,dtype=np.uint8).reshape(-1,p.line_size)[:f.height,:f.width]
                    # Equal source sample spacing; avoids RGB conversion and full-image rescaling.
                    yy=((np.arange(72)+.5)*f.height/72).astype(int)
                    xx=((np.arange(128)+.5)*f.width/128).astype(int)
                    a=a[np.ix_(yy,xx)]
                if count==0:print('FORMAT',device,mode,f.format.name,flush=True)
                a=a.astype(np.float32);count+=1
        print(json.dumps(dict(device=device,mode=mode,frames=count,wall_s=time.time()-t)),flush=True)
