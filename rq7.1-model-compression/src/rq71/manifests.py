"""Portable, immutable manifests of an existing split; never resplit images."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from .table_io import materialize_csv

SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def prepare_manifests(data_dir, manifest_dir):
    root, destination = Path(data_dir).resolve(), Path(manifest_dir)
    rows, mapping, hashes = {}, None, {}
    for split in ('train', 'val', 'test'):
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f'Missing split directory: {folder}')
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        if mapping is None:
            mapping = {name: index for index, name in enumerate(classes)}
        if classes != list(mapping):
            raise ValueError(f'Class mismatch in {split}')
        rows[split] = []
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            label = path.relative_to(folder).parts[0]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f'Cross-split exact duplicate: {relative}')
            hashes[digest] = split
            rows[split].append(dict(path=relative, class_name=label,
                class_index=mapping[label], species=label.split('__')[0],
                sha256=digest, leaf_group=''))
        if not rows[split]:
            raise ValueError(f'Empty split: {split}')
    files = {'class_to_idx.json': json.dumps(mapping, indent=2) + '\n'}
    import io
    for split, records in rows.items():
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
        files[f'{split}.csv'] = stream.getvalue()
    protocol = dict(strategy='preserve existing audited directory partitions',
        counts={s: len(r) for s, r in rows.items()}, classes=len(mapping),
        split_seed=None, leaf_group_status='unavailable; not inferred from filenames',
        limitations='Exact hash checks do not establish leaf-group or near-duplicate independence.',
        files_sha256={n: hashlib.sha256(v.encode()).hexdigest() for n, v in files.items()})
    files['split_protocol.json'] = json.dumps(protocol, indent=2) + '\n'
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = destination / name
        if target.exists() and target.read_text(encoding='utf-8') != content:
            raise ValueError(f'Manifest changed; use a new manifest directory: {target}')
    for name, content in files.items():
        target = destination / name
        if not target.exists():
            target.write_text(content, encoding='utf-8', newline='')
    return protocol


def read_manifest(data_dir, manifest_dir, split):
    root, folder = Path(data_dir).resolve(), Path(manifest_dir)
    materialize_csv(folder / f'{split}.csv')
    protocol = json.loads((folder / 'split_protocol.json').read_text(encoding='utf-8'))
    for name in ('class_to_idx.json', f'{split}.csv'):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != protocol['files_sha256'][name]:
            raise ValueError(f'Manifest integrity check failed: {name}')
    mapping = json.loads((folder / 'class_to_idx.json').read_text(encoding='utf-8'))
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Class indices must be contiguous')
    with (folder / f'{split}.csv').open(encoding='utf-8', newline='') as handle:
        records = list(csv.DictReader(handle))
    samples = []
    for row in records:
        relative = Path(row['path'])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or relative.parts[0] != split:
            raise ValueError(f'Invalid manifest path: {relative}')
        if not path.is_file() or mapping[row['class_name']] != int(row['class_index']):
            raise ValueError(f'Invalid manifest record: {relative}')
        samples.append((str(path), int(row['class_index'])))
    if not samples:
        raise ValueError(f'Empty manifest: {split}')
    return samples, mapping
