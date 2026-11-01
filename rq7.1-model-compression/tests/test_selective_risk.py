import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from selective_risk import risk_coverage

class Tests(unittest.TestCase):
    def test_perfect(self):
        r=risk_coverage([.9,.8],[0,0]); self.assertEqual(r['aurc_right_endpoint'],0)
    def test_known(self):
        r=risk_coverage([.9,.8],[0,1]); self.assertAlmostEqual(r['aurc_right_endpoint'],.25)
        self.assertEqual(r['points'][-1]['risk'],.5)
    def test_ties_order_invariant(self):
        a=risk_coverage([.9,.9],[0,1]); b=risk_coverage([.9,.9],[1,0])
        self.assertEqual(a,b); self.assertEqual(len(a['points']),1)
        self.assertEqual(a['aurc_right_endpoint'],.5)
    def test_invalid(self):
        with self.assertRaises(ValueError):risk_coverage([],[])

if __name__=='__main__':unittest.main()
