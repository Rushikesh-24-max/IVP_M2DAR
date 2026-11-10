import unittest
from summarize_test_results import calculate
class Summary(unittest.TestCase):
 def test_perfect_classification_and_cluster_intervals(self):
  rows=[dict(driver=d,label=c,probabilities=[float(i==c) for i in range(16)]) for d in ['a','b'] for c in range(16)]
  m=calculate(rows,replicates=50)
  self.assertEqual(m['samples'],32);self.assertEqual(m['accuracy'],1);self.assertEqual(m['macro_f1'],1);self.assertEqual(m['bootstrap']['accuracy_95ci'],[1,1])
 def test_macro_f1_is_not_accuracy(self):
  rows=[dict(driver=d,label=c,probabilities=[float(i==0) for i in range(16)]) for d in ['a','b'] for c in range(16)]
  m=calculate(rows,replicates=50)
  self.assertEqual(m['accuracy'],1/16);self.assertAlmostEqual(m['macro_f1'],(2/17)/16)
  self.assertEqual(m['top5_accuracy'],5/16)
if __name__=='__main__':unittest.main()
