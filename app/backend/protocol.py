"""Shared service invariants; no ML dependencies."""
from __future__ import annotations
import contextlib, hashlib, json, math, os, uuid
from pathlib import Path

CHECKPOINT='f2a254dca49cd5156b344d3abff40e21061bb3fc4e605ba1728925d2d3db3115'
SOURCE_LOCK='dc661ec1af9077855af89cca59a5aaf255910f24a5c3a02eb7ae08a08bad532c'
DATASET='08e2a5fa03d88f54600ad2e696f039bf3297cbae2b01c5ba273a54b8e2b16409'
VIEWS=('dashboard','rearview','right_side_window')
MAX_FILE=512*1024**2
ACTIVE={'uploading','receiving','validating','ready','queued','running','cancelling'}
TERMINAL={'completed','failed','cancelled'}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('w',encoding='utf-8') as f:
   json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
  tmp.replace(path)
 finally:
  with contextlib.suppress(FileNotFoundError):tmp.unlink()
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
 return h.hexdigest()
def vector(p):
 if not isinstance(p,list) or len(p)!=16 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in p) or abs(sum(p)-1)>.003:raise ValueError('Invalid model probability vector')
 return p
def validate_request(body):
 views=body.get('views')
 if not isinstance(views,list) or len(views)!=1 or views[0] not in VIEWS:raise ValueError('Exactly one supported camera viewpoint is required')
 offsets=body.get('offsets',{})
 if not isinstance(offsets,dict) or set(offsets)-set(views) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v!=0 for v in offsets.values()):raise ValueError('A single video must have zero offset')
 mode=body.get('inference_mode','sliding_window')
 if mode not in ('clip_center','sliding_window'):raise ValueError('Unknown inference mode')
 try:request_id=uuid.UUID(body['client_request_id']).hex
 except (KeyError,TypeError,ValueError,AttributeError):raise ValueError('A UUID client_request_id is required')
 title=body.get('title','Uploaded video')
 if not isinstance(title,str) or not title.strip() or len(title)>120:raise ValueError('Title must be 1 to 120 characters')
 return dict(client_request_id=request_id,title=title,views=views,offsets={views[0]:0},inference_mode=mode)

class FileLock:
 """Never unlink a lock file: replacing its inode would defeat mutual exclusion."""
 def __init__(self,path):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);self.file=path.open('a+b')
  try:
   if os.name=='nt':
    import msvcrt
    self.file.seek(0);self.file.write(b'0');self.file.flush();self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
   else:
    import fcntl
    fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except OSError:self.file.close();raise RuntimeError('Another inference service or worker holds this lock')
 def close(self):
  if self.file.closed:return
  if os.name=='nt':
   import msvcrt
   self.file.seek(0);msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
  else:
   import fcntl
   fcntl.flock(self.file,fcntl.LOCK_UN)
  self.file.close()
 def __enter__(self):return self
 def __exit__(self,*_):self.close()
