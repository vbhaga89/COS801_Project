import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import tempfile
import unittest
from run_stage import load_stage_config


class StageConfigTests(unittest.TestCase):
    def test_subprocess_config_path_is_absolute(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'seed14.json'
            path.write_text(json.dumps(dict(seed=14,batch_size=64)))
            cfg=load_stage_config(path)
            self.assertEqual(cfg['_config_path'],str(path.resolve()))
            self.assertTrue(Path(cfg['_config_path']).is_file())
            self.assertEqual(cfg['batch_size'],64)
            self.assertNotIn('_config_path',json.loads(path.read_text()))


if __name__=='__main__':unittest.main()
