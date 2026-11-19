import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from run_three_seeds import SEEDS, BATCH_SIZE, validate_config


class ScheduleTests(unittest.TestCase):
    def test_official_schedule(self):
        self.assertEqual(SEEDS, (14, 27, 42))
        self.assertEqual(BATCH_SIZE, 64)

    def test_baseline_epochs_and_recovery_budget(self):
        import json
        from pathlib import Path
        config = json.loads((Path(__file__).parents[1] / 'configs/full.json').read_text())
        self.assertEqual(config['epochs'], 20)
        self.assertEqual(config['linear_head_epochs'], 3)
        self.assertEqual(config['fine_tune_epochs'], 5)

    def test_full_config(self):
        validate_config(dict(batch_size=64, max_train_batches=None, max_eval_batches=None))

    def test_reject_changed_batch(self):
        for batch in (None, 32, 128):
            with self.assertRaisesRegex(ValueError, '64'):
                validate_config(dict(batch_size=batch))

    def test_reject_smoke(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            validate_config(dict(batch_size=64, max_train_batches=1))


if __name__ == '__main__':
    unittest.main()
