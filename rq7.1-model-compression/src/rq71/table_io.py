"""Reconstruct runtime CSV inputs from versioned, lossless JSON tables."""
import csv
import hashlib
import io
import json
from pathlib import Path


def materialize_csv(path):
    path = Path(path)
    snapshot = path.with_suffix('.json')
    if path.exists() or not snapshot.exists():
        return path
    table = json.loads(snapshot.read_text(encoding='utf-8'))
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator=table['lineterminator']).writerows(
        [table['columns'], *table['rows']])
    body = stream.getvalue().encode('utf-8')
    if hashlib.sha256(body).hexdigest() != table['csv_sha256']:
        raise ValueError(f'JSON table does not reproduce reviewed CSV: {snapshot}')
    # Never replace an existing runtime manifest.
    try:
        with path.open('xb') as handle:
            handle.write(body)
    except FileExistsError:
        if path.read_bytes() != body:
            raise ValueError(f'Conflicting runtime manifest: {path}')
    return path
