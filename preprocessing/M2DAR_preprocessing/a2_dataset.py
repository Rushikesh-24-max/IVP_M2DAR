"""Unlabelled synchronized 16-frame A2 windows with original reference times."""
from pathlib import Path
import json,sys
try:import av
except ImportError:
    sys.path.insert(0,r'F:\A1_analysis\tools');import av
import numpy as np
VIEWS=('dashboard','rearview','right_side_window')

class A2AlignedWindowDataset:
    def __init__(self,root=r'F:\A2_preprocessed',stride=16,size=448,as_torch=False):
        self.root=Path(root);self.size=int(size);self.as_torch=as_torch
        if stride<1:raise ValueError('stride must be positive')
        self.groups=[json.loads(line) for line in (self.root/'inference_groups.jsonl').read_text().splitlines() if line.strip()]
        self.windows=[]
        for i,g in enumerate(self.groups):
            n=g['frame_count'];last=max(0,n-64);starts=list(range(0,last+1,stride))
            if not starts or starts[-1]!=last:starts.append(last)
            self.windows.extend((i,start) for start in starts)

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        gi,start=self.windows[index];g=self.groups[gi]
        # The original inference wrapper samples linspace(start,start+64,16).
        indices=np.clip(np.linspace(start,start+64,16),0,g['frame_count']-1).astype(np.int64)
        wanted=set(int(i) for i in indices);images=[]
        for view in VIEWS:
            frames={};path=self.root/g['views'][view]['relative_output_path']
            with av.open(str(path)) as c:
                s=c.streams.video[0];s.codec_context.thread_count=1
                c.seek(max(0,int((indices[0]/30-.1)/s.time_base)),stream=s,backward=True)
                for f in c.decode(s):
                    i=int(round(float(f.time)*30))
                    if i in wanted:frames[i]=f.reformat(width=self.size,height=self.size,format='rgb24').to_ndarray()
                    if i>=int(indices[-1]):break
            if wanted-set(frames):raise RuntimeError('Missing PTS samples: '+str(path))
            images.append(np.stack([frames[int(i)] for i in indices]).transpose(3,0,1,2))
        video=np.ascontiguousarray((np.stack(images).astype(np.float32)/255-.45)/.225)
        if self.as_torch:
            import torch
            video=torch.from_numpy(video)
        return dict(video=video,group_id=g['group_id'],video_id=g['video_id'],driver_id=g['driver'],view_order=VIEWS,
            frame_indices=indices,reference_start_s=g['reference_start_s']+start/30,
            reference_end_s=min(g['reference_end_s'],g['reference_start_s']+(start+64)/30),
            reference_sample_times_s=g['reference_start_s']+indices/30)
