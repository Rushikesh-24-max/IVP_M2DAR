import importlib.util,json,sys,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
spec=importlib.util.spec_from_file_location('m2dar_api',Path(__file__).resolve().parents[1]/'backend/server.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class API(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.service=mod.Service(cls.tmp.name,demo_only=True,token='test-token');cls.server=mod.make_server(cls.service,port=0);cls.url='http://127.0.0.1:'+str(cls.server.server_port);threading.Thread(target=cls.server.serve_forever,daemon=True).start()
 @classmethod
 def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.service.close();cls.tmp.cleanup()
 def request(self,path,method='GET',data=None,token='test-token',origin=None):
  headers={}
  if token:headers['Authorization']='Bearer '+token
  if origin:headers['Origin']=origin
  if data is not None:data=json.dumps(data).encode();headers['Content-Type']='application/json'
  try:r=urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(self.url+path,data=data,method=method,headers=headers))
  except urllib.error.HTTPError as e:r=e
  return r.status,r.read()
 def test_auth_required(self):self.assertEqual(self.request('/api/health',token='')[0],401)
 def test_health_preview_is_explicit(self):
  code,data=self.request('/api/health');self.assertEqual(code,200);self.assertFalse(json.loads(data)['inference_available'])
 def test_unknown_origin_blocked(self):self.assertEqual(self.request('/api/health',origin='https://untrusted.example')[0],403)
 def test_frontend_and_data(self):
  self.assertEqual(self.request('/')[0],200);self.assertEqual(self.request('/data/samples.json')[0],200)
 def test_source_and_jobs_not_served_as_assets(self):
  self.assertEqual(self.request('/../backend/server.py')[0],404);self.assertEqual(self.request('/.runtime/jobs/job.json')[0],404)
 def test_preview_rejects_fake_inference(self):self.assertEqual(self.request('/api/jobs','POST',{'views':['dashboard']})[0],503)
 def test_unknown_job_404(self):self.assertEqual(self.request('/api/jobs/'+'a'*32)[0],404)
if __name__=='__main__':unittest.main()
