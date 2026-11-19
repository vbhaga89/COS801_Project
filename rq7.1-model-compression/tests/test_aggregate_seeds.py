import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import json
from pathlib import Path
import tempfile
import unittest
from aggregate_seeds import summarise


class AggregateTests(unittest.TestCase):
    def test_summary_and_protocol_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);splits=root/'splits';splits.mkdir()
            (splits/'split_protocol.json').write_text(json.dumps({'files_sha256':{'train.csv':'a'}}))
            directories=[]
            for seed in (14,27,42):
                dest=root/f'seed_{seed}';dest.mkdir();directories.append(dest)
                (dest/'full_run_manifest.json').write_text(json.dumps(dict(seed=seed,status='complete')))
                (root/f'config_seed{seed}.json').write_text(json.dumps(dict(seed=seed,batch_size=64,manifest_dir=str(splits))))
                row=dict(seed=seed,model='m',variant='baseline',teacher_model='',compression='fp32',accuracy=.9,macro_f1=.8)
                with (dest/'final_comparison.csv').open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            result=summarise(directories)
            self.assertEqual(result['summary'][0]['n'],3)
            self.assertEqual(result['summary'][0]['accuracy']['sample_sd'],0)
            (root/'config_seed42.json').write_text(json.dumps(dict(seed=42,batch_size=32,manifest_dir=str(splits))))
            with self.assertRaisesRegex(ValueError,'mismatch'):summarise(directories)


if __name__=='__main__':unittest.main()
