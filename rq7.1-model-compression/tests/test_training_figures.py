import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import tempfile
import unittest
from report_training_figures import render


class FigureTests(unittest.TestCase):
    def test_render_histories(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            history=[dict(epoch=1,stage='head',train_loss=.8,loss=.7),dict(epoch=1,stage='fine',train_loss=.5,loss=.6)]
            for model in ('mobilenet_v3_small','efficientnet_b0'):
                dest=root/model
                dest.mkdir()
                for name in ['baseline_history.json']+[f'pruning_history_r{r:.2f}.json' for r in (.1,.2,.3)]:
                    (dest/name).write_text(json.dumps(history))
                student=root/'mobilenet_v3_small_050'/f'teacher_{model}'
                student.mkdir(parents=True)
                for name in ('student_control_history.json','distillation_history.json'):
                    (student/name).write_text(json.dumps(history))
            render(root)
            self.assertEqual(len(list((root/'figures').glob('*'))),8)
            self.assertTrue(all(p.stat().st_size>1000 for p in (root/'figures').iterdir()))


if __name__=='__main__':unittest.main()
