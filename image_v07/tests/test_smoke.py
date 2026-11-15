import unittest
from classifier import combine_scores, triage

class SmokeTest(unittest.TestCase):
    def test_frozen_fusion_weight(self):
        self.assertAlmostEqual(combine_scores(0.2, 0.7), 0.6)
        self.assertEqual(combine_scores(0.42, None), 0.42)

    def test_triage_conservative(self):
        self.assertEqual(triage(.9, .8, .82), '疑似詐騙')
        self.assertIn('不代表安全', triage(.2, .1, .12))
        self.assertEqual(triage(.9, .1, .26), '需要人工確認')
        self.assertEqual(triage(.9, None, .9), '需要人工確認')
        self.assertEqual(triage(.52, .51, .512), '需要人工確認')

if __name__ == '__main__':
    unittest.main()
