import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evaluate_synchronized_test import FrameRangeReader
from verify_sync import plan_groups,module,BASE
class Reader:
 def __len__(self):return 100
 def get_avg_fps(self):return 30
 def get_batch(self,indices):return list(indices)
class Sync(unittest.TestCase):
 def test_range_reader_exact_indices(self):
  v=FrameRangeReader(Reader(),17,81)
  self.assertEqual(len(v),64);self.assertEqual(v.get_batch([0,4,63]),[17,21,80]);self.assertEqual(v.get_avg_fps(),30)
  with self.assertRaises(ValueError):v.get_batch([64])
 def test_common_crop_offset_sign_and_bounds(self):
  rows=[dict(driver='new',session='5',view=v,label=1,sample_id=v,path=v+'.mp4',sha256='hash',frames=584,source_start_s=8/30,original_label_start_s=0,original_label_end_s=20) for v in ['dashboard','rearview','right_side_window']]
  def seg(offset):return dict(reference_start_s=0,reference_end_s=20,offset_s=offset)
  mapping=dict(groups=[dict(driver='new',session='5',views={'dashboard':{},'rearview':{'segments':[seg(.3)]},'right_side_window':{'segments':[seg(-.3)]}})])
  with patch('verify_sync.read',return_value=mapping):groups,_=plan_groups(rows)
  self.assertEqual(len(groups),1);g=groups[0]
  self.assertEqual(g['views']['rearview']['start_frame']-g['views']['dashboard']['start_frame'],9)
  self.assertEqual(g['views']['right_side_window']['start_frame']-g['views']['dashboard']['start_frame'],-9)
  for v in g['views'].values():self.assertGreaterEqual(v['start_frame'],0);self.assertLessEqual(v['end_frame'],584);self.assertEqual(v['end_frame']-v['start_frame'],g['frame_count'])
 def test_repeated_correspondence_rejected(self):
  r=dict(driver='new',session='5',view='dashboard',label=1)
  with self.assertRaisesRegex(ValueError,'Ambiguous'):plan_groups([r,r])
 def test_existing_motion_sign_and_change_checks(self):
  original=module('test_existing_motion_rules',BASE/'test_alignment.py')
  with tempfile.TemporaryDirectory() as tmp:
   original.OUT=Path(tmp);original.run();checks=json.loads((Path(tmp)/'alignment_test_results.json').read_text());self.assertTrue(all(c['passed'] for c in checks))
if __name__=='__main__':unittest.main()
