"""Optional synchronized NumPy/PyTorch loader for the prepared three-view clips.

M2DAR itself can use m2dar/train.csv and val.csv with its existing Aicity loader.
This helper supplies all cameras together with identical temporal sample indices.
"""
from pathlib import Path
import json,random,sys
try:import av
except ImportError:
    sys.path.insert(0,r'F:\A1_analysis\tools')
    import av
import numpy as np
VIEWS=('dashboard','rearview','right_side_window')

class SynchronizedActionDataset:
    def __init__(self,root,split='train',num_frames=16,sampling_rate=4,size=448,normalize=True,as_torch=False,seed=42):
        if split not in ('train','val'):raise ValueError('split must be train or val')
        self.root=Path(root);self.split=split;self.num_frames=num_frames;self.sampling_rate=sampling_rate
        self.size=size;self.normalize=normalize;self.as_torch=as_torch;self.seed=seed;self.epoch=0
        self.groups=[json.loads(line) for line in (self.root/f'{split}_groups.jsonl').read_text().splitlines() if line.strip()]
        if not self.groups:raise ValueError('Empty dataset')

    def __len__(self):return len(self.groups)
    def set_epoch(self,epoch):self.epoch=int(epoch)

    def _indices(self,index,count):
        support=self.num_frames*self.sampling_rate;delta=max(count-support,0)
        start=random.Random(self.seed+self.epoch*len(self)+index).randint(0,delta) if self.split=='train' else delta//2
        # Match the baseline's uniformly sampled support and bounded indices.
        return np.clip(np.linspace(start,start+support-1,self.num_frames),0,count-1).astype(np.int64)

    def __getitem__(self,index):
        g=self.groups[index];indices=self._indices(index,g['frame_count']);wanted=set(int(x) for x in indices)
        clips=[]
        for view in VIEWS:
            frames={};path=self.root/g['views'][view]['relative_output_path']
            with av.open(str(path)) as c:
                s=c.streams.video[0];s.codec_context.thread_count=1
                c.seek(max(0,int((int(indices[0])/30-0.1)/s.time_base)),stream=s,backward=True)
                for f in c.decode(s):
                    i=int(round(float(f.time)*30))
                    if i in wanted:frames[i]=f.reformat(width=self.size,height=self.size,format='rgb24').to_ndarray()
                    if i>=int(indices[-1]):break
            missing=wanted-set(frames)
            if missing:raise RuntimeError(f'Missing prepared frame indices {missing} in {path}')
            a=np.stack([frames[int(i)] for i in indices],axis=0).transpose(3,0,1,2)
            clips.append(a)
        video=np.ascontiguousarray(np.stack(clips,axis=0)) # M,C,T,H,W
        if self.normalize:video=(video.astype(np.float32)/255.0-0.45)/0.225
        if self.as_torch:
            import torch
            video=torch.from_numpy(video)
        return dict(video=video,label=int(g['label']),group_id=g['group_id'],driver_id=g['driver'],
                    view_order=VIEWS,frame_indices=indices,temporal_padding_required=g.get('temporal_padding_required',False))
