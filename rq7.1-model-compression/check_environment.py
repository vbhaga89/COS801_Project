"""Cloud preflight: fail early on broken imports, capture actual versions/hardware."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    report = dict(python=sys.version, platform=platform.platform(), packages={}, errors={})
    for module, package in [('torch','torch'), ('torchvision','torchvision'), ('numpy','numpy'),
                            ('scipy','scipy'), ('sklearn.metrics','scikit-learn'), ('pandas','pandas'),
                            ('matplotlib.pyplot','matplotlib'), ('onnx','onnx'),
                            ('onnxruntime','onnxruntime'), ('torch_pruning','torch-pruning')]:
        print(f'Checking {module}', flush=True)
        try:
            importlib.import_module(module)
            report['packages'][package] = importlib.metadata.version(package)
        except Exception as exc:
            report['errors'][module] = f'{type(exc).__name__}: {exc}'
    if 'torch' not in report['errors']:
        import torch
        report['cuda_available'] = torch.cuda.is_available()
        report['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if args.require_cuda and not report['cuda_available']:
            report['errors']['cuda'] = 'GPU required for official training; CPU fallback rejected'
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
