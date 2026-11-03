"""Pinned hybrid-v2 protocol validation; does not infer unknown leaf identities."""
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

PROTOCOL_SHA = '603566dd5124ce5cb7ddefd88bbb12cdd1c343843e212d40b6cedf39df11aacd'
ASSIGNMENTS_SHA = '28c7093e1465563240231a6bf911b496be14f13e79d29f455cfea8a0b29a2d74'
COUNTS = {'train': 38147, 'val': 5448, 'test': 10704}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value:
        raise ValueError(f'Unsafe relative path: {value}')
    return p

def validate(folder):
    folder = Path(folder)
    if digest(folder/'split_protocol.json') != PROTOCOL_SHA:
        raise ValueError('Unreviewed hybrid protocol fingerprint')
    if digest(folder/'assignments.csv') != ASSIGNMENTS_SHA:
        raise ValueError('Hybrid assignment fingerprint mismatch')
    protocol = json.loads((folder/'split_protocol.json').read_text())
    if protocol['counts'] != COUNTS or protocol['classes'] != 38:
        raise ValueError('Hybrid counts/classes mismatch')
    for name, expected in protocol['files_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError(f'Manifest fingerprint mismatch: {name}')
    if digest(folder/'test.csv') != protocol['original_manifest_sha256']['test.csv']:
        raise ValueError('Fixed test manifest changed')
    with (folder/'assignments.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    memberships = {}
    for row in rows:
        safe_relative(row['source_path']); safe_relative(row['path'])
        split = row['new_split']
        if split not in COUNTS or not row['path'].startswith(split+'/'):
            raise ValueError('Invalid split destination')
        for key in ('group_id', 'sha256', 'author_leaf_group'):
            if row[key]:
                ident = (key, row[key])
                if memberships.setdefault(ident, split) != split:
                    raise ValueError(f'Cross-split {key}')
    for split, count in COUNTS.items():
        subset = [r for r in rows if r['new_split'] == split]
        if len(subset) != count or len({r['class_name'] for r in subset}) != 38:
            raise ValueError('Assignment counts/coverage mismatch')
        with (folder/f'{split}.csv').open(newline='', encoding='utf-8') as handle:
            actual = {(r['path'], r['sha256'], r['class_name']) for r in csv.DictReader(handle)}
        if actual != {(r['path'], r['sha256'], r['class_name']) for r in subset}:
            raise ValueError('Assignment/manifest mismatch')
    return protocol, rows
