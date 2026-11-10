import io,json,sys,tempfile,threading,time,unittest,uuid
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from server import Service,APIError
from protocol import FileLock,atomic,validate_request

def request(view='dashboard',**kwargs):return dict(client_request_id=str(uuid.uuid4()),title='test',views=[view],offsets={view:0},**kwargs)
def probe(path):return {'sha256':'test','frames':30,'duration_s':1,'pts_s':[i/30 for i in range(30)]}
class SingleVideo(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.service=Service(self.tmp.name,token='test-token',readiness={'verified':True},probe_override=probe,start_worker=False)
 def tearDown(self):self.service.close();self.tmp.cleanup()
 def create(self,view='dashboard'):return self.service.create(request(view))[0]
 def test_each_camera_singleton_and_no_multicamera(self):
  for view in ['dashboard','rearview','right_side_window']:
   j=self.create(view);self.assertEqual(j['views'],[view]);self.service.cancel(j['id']);self.service.delete(j['id'])
  for views in [[],['dashboard','rearview'],['unknown']]:
   b=request();b['views']=views
   with self.assertRaises(ValueError):validate_request(b)
 def test_offsets_and_nonfinite_rejected(self):
  for offsets in [{'dashboard':1},{'dashboard':float('nan')},{'dashboard':False},{'rearview':0}]:
   b=request();b['offsets']=offsets
   with self.assertRaises(ValueError):validate_request(b)
 def test_duplicate_create_and_conflicting_payload(self):
  b=request();j,code=self.service.create(b);same,c=self.service.create(b);self.assertEqual(j['id'],same['id']);self.assertEqual(c,200)
  b['title']='different'
  with self.assertRaises(APIError) as e:self.service.create(b)
  self.assertEqual(e.exception.code,409)
 def test_concurrent_request_only_creates_one_job(self):
  b=request();ids=[]
  threads=[threading.Thread(target=lambda:ids.append(self.service.create(b)[0]['id'])) for _ in range(8)]
  for t in threads:t.start()
  for t in threads:t.join()
  self.assertEqual(len(ids),8);self.assertEqual(len(set(ids)),1)
 def test_admission_and_idempotent_start(self):
  j=self.create();jid=j['id']
  with self.assertRaises(APIError):self.create()
  with self.assertRaises(APIError):self.service.start(jid)
  self.service.upload(jid,'dashboard',io.BytesIO(b'abc'),3)
  self.assertEqual(self.service.start(jid)['status'],'queued');self.assertEqual(self.service.start(jid)['status'],'queued');self.assertEqual(self.service.tasks.qsize(),1)
 def test_atomic_upload_and_retry_after_validation_failure(self):
  j=self.create('rearview');jid=j['id'];self.service.upload(jid,'rearview',io.BytesIO(b'abc'),3)
  self.assertEqual(self.service.read(jid)['status'],'ready');self.assertEqual((self.service.folder(jid)/'video.mp4').read_bytes(),b'abc')
  with self.assertRaises(APIError):self.service.upload(jid,'rearview',io.BytesIO(b'abc'),3)
 def test_short_upload_fails_without_committed_video(self):
  jid=self.create()['id']
  with self.assertRaises(APIError):self.service.upload(jid,'dashboard',io.BytesIO(b'x'),5)
  self.assertEqual(self.service.read(jid)['status'],'failed');self.assertFalse((self.service.folder(jid)/'video.mp4').exists());self.assertFalse((self.service.folder(jid)/'video.upload').exists())
 def test_wrong_view_and_oversize_do_not_accept_bytes(self):
  jid=self.create()['id']
  with self.assertRaises(APIError):self.service.upload(jid,'rearview',io.BytesIO(b'x'),1)
  with self.assertRaises(APIError):self.service.upload(jid,'dashboard',io.BytesIO(b'x'),513*1024**2)
  self.assertEqual(self.service.read(jid)['status'],'uploading')
 def test_cancel_upload_and_delete_only_terminal(self):
  jid=self.create()['id']
  with self.assertRaises(APIError):self.service.delete(jid)
  self.service.cancel(jid);self.assertEqual(self.service.read(jid)['status'],'cancelled');self.assertTrue(self.service.delete(jid)['deleted'])
 def test_deleted_request_id_remains_tombstoned(self):
  b=request();j,_=self.service.create(b);self.service.cancel(j['id']);self.service.delete(j['id'])
  with self.assertRaises(APIError) as e:self.service.create(b)
  self.assertEqual(e.exception.code,410)
 def test_process_lock_and_restarted_job(self):
  j=self.create();jid=j['id'];j['status']='running';self.service.save(j)
  with self.assertRaises(RuntimeError):Service(self.tmp.name,start_worker=False)
  self.service.close();self.service=Service(self.tmp.name,readiness={'verified':True},start_worker=False)
  self.assertEqual(self.service.read(jid)['status'],'failed')
 def test_unverified_service_never_advertises_gpu(self):
  self.service.readiness={'verified':False};self.assertFalse(self.service.health()['inference_available'])
  with self.assertRaises(APIError):self.create()
 def test_cancelled_job_cannot_spawn_in_dispatch_race(self):
  jid=self.create()['id'];self.service.cancel(jid)
  with patch('server.subprocess.Popen',side_effect=AssertionError('Cancelled job spawned')) as spawn:
   with self.assertRaises(APIError):self.service.owned_process(jid,[sys.executable],'test.log',1)
   spawn.assert_not_called()
 def test_request_index_recovered_from_committed_job(self):
  b=request();j,_=self.service.create(b);self.service.close();Path(self.tmp.name,'requests.json').unlink()
  self.service=Service(self.tmp.name,readiness={'verified':True},start_worker=False)
  duplicate,status=self.service.create(b);self.assertEqual(duplicate['id'],j['id']);self.assertEqual(status,200)
 def test_private_job_paths(self):
  for jid in ['../outside','a'*31,'a'*32+'/../bad']:
   with self.assertRaises(APIError):self.service.folder(jid)
 def test_retention_cleanup_leaves_active_job(self):
  jid=self.create()['id'];j=self.service.read(jid);j['status']='failed';self.service.save(j)
  j=self.service.read(jid);j['updated']=time.time()-90000;atomic(self.service.folder(jid)/'job.json',j);self.service.cleanup();self.assertFalse(self.service.folder(jid).exists())
if __name__=='__main__':unittest.main()
