import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import math
import unittest
from evaluate_saved_logits import analyse, evaluate, validate

def sample():
    return dict(class_names=['Apple healthy','Tomato diseased'],class_crops=['Apple','Tomato'],
                provenance=dict(model='synthetic test only',seed=14),
                validation=[dict(id='v0',label=0,logits=[2.,0.]),dict(id='v1',label=1,logits=[0.,2.])],
                test=[dict(id='t0',label=0,logits=[math.log(9),0.]),dict(id='t1',label=1,logits=[0.,math.log(9)])])

class Tests(unittest.TestCase):
    def test_known_calibration(self):
        d=sample(); result=evaluate(d['test'],d['class_names'],d['class_crops'])
        self.assertEqual(result['accuracy'],1.)
        self.assertAlmostEqual(result['ece_top_label'],.1)
        self.assertAlmostEqual(result['brier_multiclass_sum'],.02)
        self.assertAlmostEqual(result['nll'],-math.log(.9))
    def test_overlap_rejected(self):
        d=sample(); d['test'][0]['id']='v0'
        with self.assertRaisesRegex(ValueError,'overlap'):validate(d)
    def test_nonfinite_rejected(self):
        d=sample(); d['test'][0]['logits'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):validate(d)
    def test_test_labels_do_not_choose_temperature(self):
        a=sample(); b=copy.deepcopy(a)
        for row in b['test']:row['label']=1-row['label']
        self.assertEqual(analyse(a)['test_calibrated']['temperature'],analyse(b)['test_calibrated']['temperature'])
    def test_temperature_preserves_discrimination(self):
        result=analyse(sample())
        self.assertEqual(result['test_calibrated']['confusion_matrix'],result['test_uncalibrated']['confusion_matrix'])
    def test_errors_and_crop_support(self):
        d=sample(); d['test'][0]['label']=1
        result=evaluate(d['test'],d['class_names'],d['class_crops'])
        self.assertEqual(len(result['most_confident_errors']),1)
        self.assertEqual(result['worst_supported_crop_accuracy']['crop'],'Tomato')
        self.assertEqual(result['worst_supported_crop_accuracy']['support'],2)

if __name__=='__main__':unittest.main()
