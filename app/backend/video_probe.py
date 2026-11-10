"""Decode every frame and record source presentation times in a disposable CPU process."""
from pathlib import Path
import argparse,math,statistics
from protocol import atomic,sha256

def probe(path):
 import av
 times=[];width=height=None;last_duration=None
 with av.open(str(path)) as container:
  videos=list(container.streams.video)
  if len(videos)!=1:raise ValueError('Upload a file with exactly one video stream')
  stream=videos[0];stream.thread_type='SLICE';format_name=container.format.name
  if not any(x in format_name.split(',') for x in ['mov','mp4','matroska','webm']):raise ValueError('Supported containers: MP4, MOV and WebM')
  for frame in container.decode(stream):
   if frame.is_corrupt or frame.pts is None or frame.time_base is None:raise ValueError('Corrupt frame or missing presentation timestamp')
   if width is None:width,height=frame.width,frame.height
   if (frame.width,frame.height)!=(width,height) or not 32<=width<=3840 or not 32<=height<=2160:raise ValueError('Video dimensions must be fixed and between 32px and 3840x2160')
   t=float(frame.pts*frame.time_base)
   if not math.isfinite(t) or times and t<=times[-1]:raise ValueError('Video presentation timestamps must strictly increase')
   times.append(t)
   if t-times[0]>300 or len(times)>72000:raise ValueError('Video exceeds the duration or decoded-frame limit')
   duration=getattr(frame,'duration',None)
   if duration: last_duration=float(duration*frame.time_base)
  if len(times)<16:raise ValueError('At least 16 decoded frames are required')
  steps=[b-a for a,b in zip(times,times[1:])];step=statistics.median(steps)
  if not 1/240<=step<=1:raise ValueError('Unsupported frame rate')
  tail=last_duration if last_duration and 0<last_duration<=1 else step
  duration=times[-1]-times[0]+tail
  if not 0<duration<=300.001:raise ValueError('Maximum duration is 300 seconds')
  origin=times[0];times=[t-origin for t in times]
 gaps=[{'start_s':a+step,'end_s':b} for a,b in zip(times,times[1:]) if b-a>max(.25,5*step)]
 return dict(sha256=sha256(path),frames=len(times),width=width,height=height,duration_s=duration,pts_s=times,source_first_pts_s=origin,median_frame_interval_s=step,unsupported_intervals=gaps,constant_frame_rate=max(abs(x-step) for x in steps)<=max(.00001,step*.005),container=format_name,timing_policy='Decoded source PTS, normalized to first frame; end includes decoded/median final frame duration')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('output',type=Path);a=p.parse_args();atomic(a.output,probe(a.video))
