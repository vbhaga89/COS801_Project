"""Prospective release checks. No ML imports, training, or external writes."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    errors = []
    for path in list(ROOT.glob('*.py')) + list((ROOT / 'src/rq71').glob('*.py')):
        try:
            ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError as exc:
            errors.append(f'{path.name}: {exc}')
    cfg = json.loads((ROOT / 'configs/full.json').read_text())
    for key, value in {'batch_size': 64, 'seed': 14, 'epochs': 20,
                       'max_train_batches': None, 'max_eval_batches': None}.items():
        if cfg.get(key) != value:
            errors.append(f'Unexpected {key}: {cfg.get(key)}')
    notebooks = [ROOT/'notebooks'/name for name in ['01_dataset_setup_eda.ipynb','02_train_baselines.ipynb','03_compression.ipynb','04_final_evaluation.ipynb']]
    for cell in [c for path in notebooks for c in json.loads(path.read_text())['cells']]:
        if cell['cell_type'] == 'code':
            try:
                ast.parse(''.join(cell['source']))
            except SyntaxError as exc:
                errors.append(f'Notebook syntax: {exc}')
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append('Release notebook contains executed outputs')
    for name in ['linear_probe.py', 'run_three_seeds.py', 'check_environment.py',
                 'src/rq71/manifests.py', 'ENVIRONMENT.md']:
        if not (ROOT / name).is_file():
            errors.append(f'Missing {name}')
    print('Prospective static checks:', 'FAIL' if errors else 'PASS')
    for error in errors:
        print(error)
    print('Not a full runtime certification. Read ENVIRONMENT.md and the release review before official training.')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
