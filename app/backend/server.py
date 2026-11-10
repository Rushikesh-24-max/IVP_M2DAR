"""Authenticated, loopback-first single-video inference service."""
from __future__ import annotations
import argparse,contextlib,hmac,json,mimetypes,os,queue,re,shutil,subprocess,sys,threading,time,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,unquote
sys.path.insert(0,str(Path(__file__).resolve().parent))
from protocol import ACTIVE,TERMINAL,VIEWS,MAX_FILE,FileLock,atomic,read,validate_request
ROOT=Path(__file__).resolve().parents[1]
JOB_ID=re.compile(r'^[a-f0-9]{32}$')

class APIError(Exception):
 def __init__(self,code,message):self.code=code;super().__init__(message)

class Service:
 def __init__(self,root,training_root=None,bundle=None,checkpoint=None,demo_only=False,token='',origins=(),readiness=None,probe_override=None,start_worker=True):
  self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);os.chmod(self.root,0o700)
  self.instance_lock=FileLock(self.root/'.service.lock');self.lock=threading.RLock();self.stop=threading.Event()
  self.training_root=training_root;self.bundle=bundle;self.checkpoint=checkpoint;self.demo_only=demo_only;self.token=token;self.origins=set(origins)
  self.readiness=readiness or {'verified':False};self.probe_override=probe_override;self.tasks=queue.Queue(maxsize=1);self.processes={};self.receiving=set();self.thread=None
  self.requests=read(self.root/'requests.json') if (self.root/'requests.json').exists() else {}
  for p in self.root.glob('*/job.json'):
   j=read(p)
   key=j.get('client_request_id')
   if key and key not in self.requests:self.requests[key]={'id':j['id'],'spec':{k:j[k] for k in ['client_request_id','title','views','offsets','inference_mode']},'created':j['created']}
   if j['status'] in ACTIVE:j.update(status='failed',error='Service restarted before this job completed. No automatic retry was scheduled.');self.save(j)
  atomic(self.root/'requests.json',self.requests)
  if start_worker:self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start()
 def close(self):
  self.stop.set()
  with self.lock:
   for jid,proc in list(self.processes.items()):
    j=self.read(jid);j.update(status='cancelling');self.save(j)
    if proc.poll() is None:proc.terminate()
  for proc in list(self.processes.values()):
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  if self.thread:self.thread.join(timeout=15)
  self.instance_lock.close()
 def folder(self,jid):
  if not JOB_ID.fullmatch(jid):raise APIError(400,'Invalid job identifier')
  p=self.root/jid
  if p.is_symlink() or not p.resolve().is_relative_to(self.root):raise APIError(400,'Unsafe job directory')
  return p
 def read(self,jid):return read(self.folder(jid)/'job.json')
 def save(self,j):
  folder=self.folder(j['id']);folder.mkdir(mode=0o700,exist_ok=True);j['updated']=time.time();atomic(folder/'job.json',j)
 def jobs(self):return [read(p) for p in self.root.glob('*/job.json')]
 def ensure_ready(self):
  if self.demo_only or not self.readiness.get('verified'):raise APIError(503,'The GPU model is not verified and ready')
 def health(self):
  with self.lock:active=[j['id'] for j in self.jobs() if j['status'] in ACTIVE]
  ready=not self.demo_only and self.readiness.get('verified',False)
  return dict(status='ready' if ready else 'preview',inference_available=ready,model_ready=ready,model=self.readiness,device=os.environ.get('CUDA_VISIBLE_DEVICES','not set'),supported_views=list(VIEWS),max_file_mb=512,max_duration_s=300,max_retained_jobs=10,retention_hours=24,storage_reserve_gib=2,active_jobs=active,max_active_jobs=1)
 def cleanup(self):
  now=time.time()
  for j in self.jobs():
   if j['status'] in {'uploading','ready'} and now-j.get('updated',j['created'])>3600:
    j.update(status='failed',error='Idle upload expired after one hour');self.save(j)
   if j['status'] in TERMINAL and now-j.get('updated',j['created'])>86400:self.delete(j['id'])
  self.requests={k:v for k,v in self.requests.items() if now-v['created']<7*86400};atomic(self.root/'requests.json',self.requests)
 def create(self,body):
  self.ensure_ready();spec=validate_request(body)
  with self.lock:
   self.cleanup();key=spec['client_request_id']
   if key in self.requests:
    old=self.requests[key]
    if old['spec']!=spec:raise APIError(409,'This request ID was already used with different parameters')
    try:return self.read(old['id']),200
    except FileNotFoundError:raise APIError(410,'This request was already processed and its job was deleted; choose a new request ID')
   if any(j['status'] in ACTIVE for j in self.jobs()):raise APIError(409,'Another video job is active; finish or cancel it first')
   if len(self.jobs())>=10 or len(self.requests)>=10000:raise APIError(409,'Job retention limit reached; delete an inactive job')
   if shutil.disk_usage(self.root).free<2*1024**3:raise APIError(507,'At least 2 GiB free storage is required')
   j=dict(spec,id=uuid.uuid4().hex,status='uploading',created=time.time(),progress=0,message='Waiting for one video',files={})
   self.save(j);self.requests[key]={'id':j['id'],'spec':spec,'created':j['created']};atomic(self.root/'requests.json',self.requests)
   return j,201
 def owned_process(self,jid,command,log_name,timeout):
  folder=self.folder(jid)
  with (folder/log_name).open('wb') as log:
   with self.lock:
    if self.read(jid)['status'] not in {'running','validating'}:raise APIError(409,'Job was cancelled or is not executable')
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);self.processes[jid]=proc
   try:
    try:return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
     proc.kill();proc.wait();raise APIError(422,'Video processing exceeded its time limit')
   finally:
    with self.lock:self.processes.pop(jid,None)
 def upload(self,jid,view,stream,n):
  self.ensure_ready()
  if not isinstance(n,int) or not 0<n<=MAX_FILE:raise APIError(413,'Video must be between 1 byte and 512 MiB')
  folder=self.folder(jid);temp=folder/'video.upload'
  with self.lock:
   j=self.read(jid)
   if j['status']!='uploading' or j['views']!=[view]:raise APIError(409,'This job is not accepting that camera upload')
   if shutil.disk_usage(self.root).free<n+2*1024**3:raise APIError(507,'Upload would exceed the 2 GiB storage reserve')
   j.update(status='receiving',message='Receiving one video');self.save(j);self.receiving.add(jid)
  try:
   with temp.open('wb') as out:
    left=n
    while left:
     chunk=stream.read(min(left,1024**2))
     if not chunk:raise APIError(400,'Incomplete upload')
     with self.lock:
      if self.read(jid)['status']=='cancelling':raise APIError(409,'Job was cancelled')
     out.write(chunk);left-=len(chunk)
   with self.lock:
    j=self.read(jid)
    if j['status']=='cancelling':raise APIError(409,'Job was cancelled')
    j.update(status='validating',message='Decoding and checking source timestamps');self.save(j)
   if self.probe_override:probe=self.probe_override(temp);atomic(folder/'probe.json',probe)
   else:
    command=[sys.executable,str(ROOT/'backend/video_probe.py'),str(temp),str(folder/'probe.json')]
    if self.owned_process(jid,command,'validation.log',180):raise APIError(422,'Video validation failed: use a decodable MP4, MOV or WebM with at least 16 frames, no corrupt frames, increasing timestamps, and a duration of at most five minutes')
    probe=read(folder/'probe.json')
   with self.lock:
    j=self.read(jid)
    if j['status']=='cancelling':raise APIError(409,'Job was cancelled')
    temp.replace(folder/'video.mp4');j.update(status='ready',message='Video validated; ready to run',files={view:{k:v for k,v in probe.items() if k!='pts_s'}});self.save(j)
   return {'uploaded':view,'status':'ready','media':j['files'][view]}
  except Exception as e:
   with self.lock:
    j=self.read(jid);cancelled=j['status']=='cancelling';j.update(status='cancelled' if cancelled else 'failed',error='Upload cancelled' if cancelled else str(e),message='Upload did not complete');self.save(j)
   raise
  finally:
   with self.lock:self.receiving.discard(jid)
   with contextlib.suppress(FileNotFoundError):temp.unlink()
 def start(self,jid):
  self.ensure_ready()
  with self.lock:
   j=self.read(jid)
   if j['status'] in {'queued','running','completed'}:return j
   if j['status']!='ready':raise APIError(409,'Upload and validate the video before running')
   j.update(status='queued',message='Waiting for the GPU worker');self.save(j)
   try:self.tasks.put_nowait(jid)
   except queue.Full:j.update(status='ready');self.save(j);raise APIError(429,'Worker queue is full')
   return j
 def cancel(self,jid):
  with self.lock:
   j=self.read(jid)
   if j['status'] in TERMINAL:return j
   proc=self.processes.get(jid);active_io=jid in self.receiving
   j.update(status='cancelling' if proc or active_io else 'cancelled',message='Cancellation requested');self.save(j)
   if proc and proc.poll() is None:
    proc.terminate()
    def escalate():
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:
      if proc.poll() is None:proc.kill()
    threading.Thread(target=escalate,daemon=True).start()
   return j
 def delete(self,jid):
  with self.lock:
   j=self.read(jid)
   if j['status'] in ACTIVE or jid in self.receiving or jid in self.processes:raise APIError(409,'Cancel or finish the job before deletion')
   folder=self.folder(jid)
   if folder.parent!=self.root or not JOB_ID.fullmatch(folder.name):raise APIError(400,'Unsafe deletion target')
   shutil.rmtree(folder)
  return {'deleted':True}
 def worker(self):
  last_cleanup=time.monotonic()
  while not self.stop.is_set():
   try:jid=self.tasks.get(timeout=.25)
   except queue.Empty:
    if time.monotonic()-last_cleanup>60:
     with self.lock:self.cleanup()
     last_cleanup=time.monotonic()
    continue
   try:
    with self.lock:
     j=self.read(jid)
     if j['status']!='queued':continue
     j.update(status='running',message='Loading the frozen checkpoint');self.save(j)
    command=[sys.executable,str(ROOT/'backend/gpu_worker.py'),'--job',str(self.folder(jid)/'job.json'),'--training-root',str(self.training_root),'--bundle',str(self.bundle),'--checkpoint',str(self.checkpoint)]
    code=self.owned_process(jid,command,'worker.log',3600)
    with self.lock:
     j=self.read(jid)
     if j['status'] in {'cancelling','cancelled'}:j.update(status='cancelled',message='Job cancelled')
     elif code or not (self.folder(jid)/'result.json').exists():j.update(status='failed',error='GPU inference failed. The server retained its worker log; no successful result was returned.')
     else:j.update(status='completed',progress=100,message='Prediction completed')
     self.save(j)
   except Exception as e:
    with self.lock:
     with contextlib.suppress(FileNotFoundError):
      j=self.read(jid);j.update(status='cancelled' if j['status'] in {'cancelling','cancelled'} else 'failed',error=str(e));self.save(j)
   finally:self.tasks.task_done()

class Handler(BaseHTTPRequestHandler):
 server_version='M2DAR/2.0'
 def setup(self):super().setup();self.connection.settimeout(60)
 @property
 def service(self):return self.server.service
 def log_message(self,fmt,*args):sys.stderr.write('%s %s\n'%(self.command,urlparse(self.path).path))
 def origin_ok(self):
  origin=self.headers.get('Origin');own={f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}'}
  return not origin or origin in own or origin in self.service.origins
 def headers_out(self,code,kind,size,extra=()):
  self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(size));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Cache-Control','no-store' if self.path.startswith('/api/') else 'no-cache')
  if self.headers.get('Origin') and self.origin_ok():
   self.send_header('Access-Control-Allow-Origin',self.headers['Origin']);self.send_header('Vary','Origin');self.send_header('Access-Control-Allow-Methods','GET,POST,DELETE,OPTIONS');self.send_header('Access-Control-Allow-Headers','Authorization,Content-Type')
  for k,v in extra:self.send_header(k,v)
  self.end_headers()
 def respond(self,code,value):
  data=json.dumps(value,allow_nan=False).encode();self.headers_out(code,'application/json',len(data))
  if self.command!='HEAD':self.wfile.write(data)
 def authorize(self):
  if not self.origin_ok():raise APIError(403,'Origin is not allowed')
  if self.service.token and not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+self.service.token):raise APIError(401,'Enter the backend API token')
 def body(self):
  n=int(self.headers.get('Content-Length','0'))
  if not 0<n<=65536:raise APIError(400,'Invalid JSON request size')
  b=json.loads(self.rfile.read(n))
  if not isinstance(b,dict):raise APIError(400,'Expected a JSON object')
  return b
 def do_OPTIONS(self):
  if not self.origin_ok():return self.respond(403,{'error':'Origin is not allowed'})
  self.headers_out(204,'application/json',0)
 def do_HEAD(self):self.do_GET()
 def do_GET(self):self.dispatch('GET')
 def do_POST(self):self.dispatch('POST')
 def do_DELETE(self):self.dispatch('DELETE')
 def dispatch(self,method):
  try:
   path=urlparse(self.path).path
   if not path.startswith('/api/'):
    if method!='GET':raise APIError(404,'Unknown endpoint')
    return self.static(path)
   self.authorize()
   if method=='GET' and path=='/api/health':return self.respond(200,self.service.health())
   if method=='GET' and path=='/api/jobs':
    with self.service.lock:self.service.cleanup();jobs=sorted(self.service.jobs(),key=lambda j:j['created'],reverse=True)
    return self.respond(200,{'jobs':jobs})
   if method=='POST' and path=='/api/jobs':
    j,code=self.service.create(self.body());return self.respond(code,j)
   match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(result|run|cancel|videos/(dashboard|rearview|right_side_window)))?',path)
   if not match:raise APIError(404,'Unknown endpoint')
   jid,action,view=match.groups()
   if method=='POST' and view:
    if self.headers.get('Transfer-Encoding'):raise APIError(400,'Send a bounded Content-Length upload')
    return self.respond(200,self.service.upload(jid,view,self.rfile,int(self.headers.get('Content-Length','0'))))
   if method=='POST' and action=='run':return self.respond(202,self.service.start(jid))
   if method=='POST' and action=='cancel':return self.respond(200,self.service.cancel(jid))
   if method=='DELETE' and action is None:return self.respond(200,self.service.delete(jid))
   if method=='GET':
    with self.service.lock:j=self.service.read(jid)
    if action=='result':
     if j['status']!='completed':raise APIError(409,'A completed result is not available')
     return self.respond(200,read(self.service.folder(jid)/'result.json'))
    if action is None:
     if j['status']=='running':
      with contextlib.suppress(FileNotFoundError,ValueError):j.update(read(self.service.folder(jid)/'progress.json'))
     return self.respond(200,j)
   raise APIError(404,'Unknown endpoint')
  except APIError as e:self.respond(e.code,{'error':str(e)})
  except FileNotFoundError:self.respond(404,{'error':'Job or file not found'})
  except (ValueError,TypeError,KeyError):self.respond(400,{'error':'Invalid request'})
  except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
  except Exception:self.respond(500,{'error':'Unexpected service failure; inspect the private server log'})
 def static(self,path):
  root=(ROOT/'dist').resolve();file=(root/unquote(path).lstrip('/')).resolve()
  if path=='/':file=root/'index.html'
  if not file.is_relative_to(root) or not file.is_file():raise APIError(404,'File not found')
  size=file.stat().st_size;start=0;end=size-1;code=200;extra=[('Accept-Ranges','bytes')]
  if self.headers.get('Range'):
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
   if not m:raise APIError(416,'Unsupported byte range')
   start=int(m[1]);end=min(end,int(m[2])) if m[2] else end
   if start>end:raise APIError(416,'Invalid byte range')
   code=206;extra.append(('Content-Range',f'bytes {start}-{end}/{size}'))
  self.headers_out(code,mimetypes.guess_type(file.name)[0] or 'application/octet-stream',end-start+1,extra)
  if self.command=='HEAD':return
  with file.open('rb') as f:
   f.seek(start);left=end-start+1
   while left:
    chunk=f.read(min(left,1024**2));self.wfile.write(chunk);left-=len(chunk)

def make_server(service,host='127.0.0.1',port=8765):
 s=ThreadingHTTPServer((host,port),Handler);s.daemon_threads=True;s.service=service;return s
def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765);p.add_argument('--training-root',type=Path,default=Path.home()/'m2dar-training');p.add_argument('--bundle',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--jobs-root',type=Path,default=ROOT/'.runtime/jobs');p.add_argument('--demo-only',action='store_true');a=p.parse_args()
 token=os.environ.get('M2DAR_API_TOKEN','')
 if not a.demo_only and len(token)<24:p.error('Set a private M2DAR_API_TOKEN of at least 24 characters')
 bundle=(a.bundle or a.training_root/'prepared/server_b1_amp_sdpa').resolve();checkpoint=(a.checkpoint or a.training_root/'runs/m2dar_s_40drivers_b1_amp_sdpa_seed42/best.pyth').resolve();ready=None
 if not a.demo_only:
  a.jobs_root.mkdir(parents=True,exist_ok=True);status=a.jobs_root.parent/'model_readiness.json'
  command=[sys.executable,str(ROOT/'backend/gpu_worker.py'),'--check','--training-root',str(a.training_root),'--bundle',str(bundle),'--checkpoint',str(checkpoint),'--readiness-out',str(status)]
  subprocess.run(command,check=True,timeout=180);ready=read(status)
 service=Service(a.jobs_root,a.training_root,bundle,checkpoint,a.demo_only,token,[x for x in os.environ.get('M2DAR_ALLOWED_ORIGINS','').split(',') if x],readiness=ready)
 server=make_server(service,a.host,a.port)
 import signal
 def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 print(f'M2DAR single-video service: http://{a.host}:{a.port}',flush=True)
 try:server.serve_forever()
 finally:server.server_close();service.close()
if __name__=='__main__':main()
