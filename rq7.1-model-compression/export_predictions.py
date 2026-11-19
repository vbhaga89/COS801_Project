"""Explicit selected-ONNX prediction export; no training or candidate selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    cfg=json.loads(args.config.read_text())
    if cfg.get('max_eval_batches') is not None:
        raise ValueError('Official prediction export must evaluate the entire split')
    if not cfg.get('manifest_dir'):raise ValueError('Immutable manifests required')
    sys.path.insert(0,str(Path(__file__).parent/'src'))
    from rq71.manifests import read_manifest
    from rq71.data import ManifestDataset, build_transforms
    from torch.utils.data import DataLoader
    import numpy as np
    import onnxruntime as ort
    manifest=Path(cfg['manifest_dir'])
    protocol=json.loads((manifest/'split_protocol.json').read_text())
    mapping=json.loads((manifest/'class_to_idx.json').read_text())
    names=sorted(mapping,key=mapping.get)
    _,transform=build_transforms(int(cfg['image_size']))
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(args.model),sess_options=options,providers=['CPUExecutionProvider'])
    payload=dict(class_names=names,class_crops=[n.split('__')[0] for n in names],validation=[],test=[],
                 provenance=dict(seed=cfg['seed'],artifact=str(args.model),artifact_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                                 split_hashes=protocol['files_sha256'],onnxruntime=ort.__version__,
                                 preprocessing='RGB resize to configured square; tensor; ImageNet normalization'))
    for split,target in [('val','validation'),('test','test')]:
        samples,_=read_manifest(cfg['data_dir'],manifest,split)
        # Verify evaluated bytes, not just CSV integrity.
        import csv
        with (manifest/f'{split}.csv').open(newline='',encoding='utf-8') as handle:records=list(csv.DictReader(handle))
        for (path,_),row in zip(samples,records):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'Image mismatch: {path}')
        loader=DataLoader(ManifestDataset(cfg['data_dir'],manifest,split,transform),batch_size=64,shuffle=False,num_workers=0)
        offset=0
        for images,labels in loader:
            logits=session.run(None,{session.get_inputs()[0].name:images.numpy().astype(np.float32)})[0]
            if logits.shape!=(len(labels),len(names)) or not np.isfinite(logits).all():raise ValueError('Invalid logits')
            for label,values in zip(labels.tolist(),logits.tolist()):
                row=records[offset]
                if label!=int(row['class_index']):raise ValueError('Prediction/sample-order mismatch')
                payload[target].append(dict(id=row['path'],label=label,logits=values));offset+=1
            print(f'{split}: {offset}/{len(records)}',flush=True)
        if offset!=protocol['counts'][split]:raise ValueError('Incomplete prediction export')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:json.dump(payload,handle,allow_nan=False)
    print(args.output)


if __name__=='__main__':main()
