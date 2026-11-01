import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ExportTests(unittest.TestCase):
    def test_real_onnx_export_and_stable_ids(self):
        import numpy as np
        import torch
        from PIL import Image
        sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
        from rq71.manifests import prepare_manifests
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';splits=root/'splits'
            for index,split in enumerate(('train','val','test')):
                for label in range(2):
                    dest=data/split/f'Crop{label}__healthy';dest.mkdir(parents=True)
                    Image.fromarray(np.full((8,8,3),index*50+label,dtype=np.uint8)).save(dest/'image.png')
            prepare_manifests(data,splits)
            model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*8*8,2)).eval()
            artifact=root/'model.onnx'
            torch.onnx.export(model,torch.zeros(1,3,8,8),str(artifact),input_names=['input'],output_names=['logits'],dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}},opset_version=17,dynamo=False)
            config=root/'config.json'
            config.write_text(json.dumps(dict(seed=14,data_dir=str(data),manifest_dir=str(splits),image_size=8,max_eval_batches=None)))
            out=root/'predictions.json'
            subprocess.run([sys.executable,str(Path(__file__).parents[1]/'export_predictions.py'),'--config',str(config),'--model',str(artifact),'--output',str(out)],check=True)
            result=json.loads(out.read_text())
            self.assertEqual(len(result['validation']),2)
            self.assertEqual(len(result['test']),2)
            self.assertTrue(all(r['id'].startswith('test/') for r in result['test']))
            self.assertTrue(all(len(r['logits'])==2 for r in result['test']))


if __name__=='__main__':unittest.main()
