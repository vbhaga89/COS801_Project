import inspect
import unittest
import run_three_seeds
import stage_plantvillage


class FinalSplitDefaultsTests(unittest.TestCase):
    def test_launcher_defaults_to_final_protocol(self):
        self.assertEqual(inspect.signature(run_three_seeds.verify_data).parameters['split_version'].default, 'hybrid-v2')

    def test_staging_uses_single_split_directory(self):
        self.assertIn("parent/'splits'", inspect.getsource(stage_plantvillage.main))
        self.assertNotIn('splits_hybrid_v2', inspect.getsource(stage_plantvillage.main))


if __name__ == '__main__':
    unittest.main()
