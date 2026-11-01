import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from rq71.selection import choose_pruning, PruningAccuracyGateError


class PruningGateTests(unittest.TestCase):
    def row(self,ratio,drop):
        return dict(pruning_ratio=ratio,validation_accuracy_drop=drop,validation_metrics={'macro_f1':.8})
    def test_all_ineligible(self):
        with self.assertRaises(PruningAccuracyGateError):
            choose_pruning([self.row(.1,.0556),self.row(.2,.1)],.03)
    def test_highest_eligible(self):
        self.assertEqual(choose_pruning([self.row(.1,.01),self.row(.2,.03),self.row(.3,.04)],.03)['pruning_ratio'],.2)
    def test_missing_candidates_is_not_accuracy_rejection(self):
        with self.assertRaises(ValueError):choose_pruning([],.03)


if __name__=='__main__':unittest.main()
