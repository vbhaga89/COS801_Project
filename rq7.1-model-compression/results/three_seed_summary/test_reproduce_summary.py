import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from reproduce_summary import generate

SOURCE = Path(__file__).resolve().parent / 'final_consolidation'
if not SOURCE.exists():
    SOURCE = Path(__file__).resolve().parent.parent


def read_csv(path):
    return list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for seed in [14, 27, 42]:
            shutil.copytree(SOURCE / f'RQ7_1_Model_Compression_Seed{seed}', self.root / f'RQ7_1_Model_Compression_Seed{seed}')

    def tearDown(self):
        self.temp.cleanup()

    def change_json(self, seed, name, change):
        p = self.root / f'RQ7_1_Model_Compression_Seed{seed}' / name
        obj = json.loads(p.read_text())
        change(obj)
        p.write_text(json.dumps(obj))

    def test_full_tables_and_independent_mean(self):
        generate(self.root, self.root / 'summary')
        rows = read_csv(self.root / 'summary/comparison_summary.csv')
        self.assertEqual(len(rows), 9)
        row = next(r for r in rows if r['model'] == 'efficientnet_b0' and r['variant'] == 'quantized')
        nums = [0.9783258594917787, 0.9884155455904334, 0.9837443946188341]
        mean = sum(nums) / 3
        sd = (sum((x - mean)**2 for x in nums) / 2)**0.5
        self.assertAlmostEqual(float(row['accuracy_mean']), mean, places=14)
        self.assertAlmostEqual(float(row['accuracy_sample_sd']), sd, places=14)
        self.assertEqual(len(read_csv(self.root / 'summary/calibration_by_seed.csv')), 54)
        self.assertEqual(len(read_csv(self.root / 'summary/worst_groups_by_seed.csv')), 27)

    def test_reject_configuration_change(self):
        self.change_json(27, 'config_seed27.json', lambda d: d.update(batch_size=32))
        with self.assertRaises(AssertionError):
            generate(self.root, self.root / 'summary')

    def test_reject_comparison_change(self):
        self.change_json(14, 'final_comparison.json', lambda d: d['rows'][0].__setitem__(0, 'changed'))
        with self.assertRaisesRegex(AssertionError, 'fingerprint'):
            generate(self.root, self.root / 'summary')

    def test_reject_split_change(self):
        self.change_json(42, 'split_protocol.json', lambda d: d['files_sha256'].update({'train.csv': 'wrong'}))
        with self.assertRaises(AssertionError):
            generate(self.root, self.root / 'summary')

    def test_reject_duplicate_evaluation(self):
        self.change_json(14, 'evaluation_metrics.json', lambda d: d['records'].__setitem__(1, d['records'][0]))
        with self.assertRaises(AssertionError):
            generate(self.root, self.root / 'summary')

    def test_reject_environment_difference(self):
        self.change_json(42, 'environment.json', lambda d: d['packages'].update(torch='different'))
        with self.assertRaises(AssertionError):
            generate(self.root, self.root / 'summary')


if __name__ == '__main__':
    unittest.main()
