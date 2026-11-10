"""Verify test-evaluation gates using isolated metadata fixtures, never real test predictions."""
import json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from evaluate_test import verify_test,stable,digest,write_loader_csv,EXPECTED_CHECKPOINT
from prepare_test import source_timing_evidence
class Gates(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.protocol=dict(test_drivers=['unseen'],training_drivers=['train'],validation_drivers=['val'],checkpoint_sha256=EXPECTED_CHECKPOINT)
  (self.root/'clip.bin').write_bytes(b'fixture')
  self.rows=[dict(sample_id=str(i),driver='unseen',label=i,split='test',path='clip.bin',size_bytes=7,sha256=digest(self.root/'clip.bin')) for i in range(16)]
  self.save()
 def tearDown(self):self.tmp.cleanup()
 def save(self):
  (self.root/'protocol_lock.json').write_text(json.dumps(self.protocol))
  (self.root/'test_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
  ready=dict(status='ready_for_frozen_checkpoint_evaluation',protocol_sha256=stable(self.protocol),index_sha256=digest(self.root/'test_index.jsonl'),checkpoint_sha256=EXPECTED_CHECKPOINT,verified_test_camera_clips=len(self.rows))
  (self.root/'READY.json').write_text(json.dumps(ready))
 def test_valid_fixture(self):self.assertEqual(len(verify_test(self.root)[1]),16)
 def test_loader_uses_actual_configuration_separator(self):
  file=self.root/'loader.csv'
  write_loader_csv(file,[dict(path='clips/test.mp4',label=3)],'\t')
  self.assertEqual(file.read_text(),'clips/test.mp4\t3\n')
  write_loader_csv(file,[dict(path='clips/test.mp4',label=3)],' ')
  self.assertEqual(file.read_text(),'clips/test.mp4 3\n')
 def test_no_ready_no_evaluation(self):
  (self.root/'READY.json').unlink()
  with self.assertRaises(FileNotFoundError):verify_test(self.root)
 def test_driver_leakage_rejected(self):
  self.protocol['training_drivers']=['unseen'];self.save()
  with self.assertRaisesRegex(ValueError,'leakage'):verify_test(self.root)
 def test_index_tampering_rejected(self):
  with (self.root/'test_index.jsonl').open('a') as f:f.write('\n')
  with self.assertRaisesRegex(ValueError,'index changed'):verify_test(self.root)
 def test_clip_tampering_rejected(self):
  (self.root/'clip.bin').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'media changed'):verify_test(self.root)
 def test_nonfixed_checkpoint_rejected(self):
  (self.root/'other.pyth').write_bytes(b'not epoch170')
  with self.assertRaisesRegex(ValueError,'fixed epoch-170'):verify_test(self.root,self.root/'other.pyth')
 def test_unsafe_path_rejected(self):
  self.rows[0]['path']='../escape.mp4';self.save()
  with self.assertRaisesRegex(ValueError,'unsafe'):verify_test(self.root)
 def test_average_fps_requires_matching_source_decode_audit(self):
  audit=dict(key='video',source_sha256='exact-source',corrupt_frames=0,nonincreasing_pts=0,median_frame_interval_s=1/30,max_frame_interval_s=1/30)
  (self.root/'fps_investigation.json').write_text(json.dumps([audit]))
  with patch('prepare_test.ROOT',self.root):
   self.assertIsNotNone(source_timing_evidence({'key':'video'},30.015,'exact-source'))
   with self.assertRaises(ValueError):source_timing_evidence({'key':'video'},30.015,'changed-source')
   audit['corrupt_frames']=1;(self.root/'fps_investigation.json').write_text(json.dumps([audit]))
   with self.assertRaises(ValueError):source_timing_evidence({'key':'video'},30.015,'exact-source')
if __name__=='__main__':unittest.main()
