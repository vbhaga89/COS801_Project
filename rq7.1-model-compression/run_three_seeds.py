"""Explicit, non-overwriting seed launcher. Preflight is the default; training is opt-in."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SEEDS = (14, 27, 42)
BATCH_SIZE = 64


def validate_config(config):
    if config.get('smoke_mode'):
        raise ValueError('Smoke configuration cannot run as official')
    if config.get('batch_size') != BATCH_SIZE:
        raise ValueError('Official RQ7.1 training batch_size must be 64; no automatic reduction')
    if config.get('max_train_batches') is not None or config.get('max_eval_batches') is not None:
        raise ValueError('Batch-limited/smoke configuration cannot be an official run')


def verify_data(root, manifests):
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from rq71.manifests import read_manifest
    protocol = json.loads((manifests / 'split_protocol.json').read_text())
    expected = {'train': 38146, 'val': 5449, 'test': 10704}
    if protocol['counts'] != expected or protocol['classes'] != 38:
        raise ValueError('Official run requires the audited 54,299-image, 38-class split')
    seen = {}
    for split, count in expected.items():
        samples, _ = read_manifest(root, manifests, split)
        import csv
        with (manifests / f'{split}.csv').open(newline='', encoding='utf-8') as handle:
            records = list(csv.DictReader(handle))
        if len(samples) != count:
            raise ValueError(f'Wrong count for {split}')
        for (path, _), row in zip(samples, records):
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest != row['sha256']:
                raise ValueError(f'Image checksum mismatch: {row["path"]}')
            if digest in seen and seen[digest] != split:
                raise ValueError(f'Cross-split duplicate: {row["path"]}')
            seen[digest] = split
        print(f'Verified {split}: {count} image hashes', flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/full.json')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s not in SEEDS for s in args.seeds):
        parser.error('Use distinct seeds from 14, 27, 42')
    config = json.loads(args.config.read_text())
    validate_config(config)
    verify_data(args.data_dir.resolve(), args.manifest_dir.resolve())
    print(f'Full-data preflight passed. Scheduled seeds: {args.seeds}; training batch size: {BATCH_SIZE}', flush=True)
    if not args.execute:
        print('No training started. Add --execute to launch.', flush=True)
        return
    run = args.output_root / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    subprocess.run([sys.executable, '-u', str(Path(__file__).parent / 'check_environment.py'),
                    '--require-cuda', '--output', str((run / 'environment.json').resolve())],
                   env=env, check=True)
    for seed in args.seeds:
        resolved = dict(config, seed=seed, data_dir=str(args.data_dir.resolve()),
                        manifest_dir=str(args.manifest_dir.resolve()),
                        output_dir=str((run / f'seed_{seed}').resolve()),
                        quarantine_cross_split_duplicates=False)
        path = run / f'config_seed{seed}.json'
        path.write_text(json.dumps(resolved, indent=2), encoding='utf-8')
        for model in ('mobilenet_v3_small', 'efficientnet_b0'):
            probe_cmd = [sys.executable, '-u', str(Path(__file__).parent / 'linear_probe.py'),
                         '--config', str(path.resolve()), '--model', model]
            # Inherit live output. Full runner logs remain separate from probe evidence.
            subprocess.run(probe_cmd, env=env, check=True)
        cmd = [sys.executable, '-u', str(Path(__file__).parent / 'run_experiment.py'),
               'full', '--config', str(path.resolve())]
        print(f'Starting seed {seed}; saved config: {path}', flush=True)
        with (run / f'seed_{seed}.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', env=env) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code:
            raise SystemExit(f'Seed {seed} failed with exit {code}; logs preserved, later seeds not started')
        subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'analyse_selected_models.py'),
                        '--config',str(path.resolve())],env=env,check=True)
    if set(args.seeds)==set(SEEDS):
        subprocess.run([sys.executable,str(Path(__file__).parent/'aggregate_seeds.py'),
                        *[str((run/f'seed_{seed}').resolve()) for seed in SEEDS],
                        '--output',str((run/'three_seed_summary.json').resolve())],env=env,check=True)
    print(f'Seed schedule finished; audit scientific status in each full_run_manifest.json: {run}', flush=True)


if __name__ == '__main__':
    main()
