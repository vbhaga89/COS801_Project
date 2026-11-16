import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.rq71.table_io import materialize_csv


class TableTests(unittest.TestCase):
    def test_exact_csv_bytes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            body = b'path,label\r\n"a,b",1\r\n'
            snapshot = dict(columns=['path', 'label'], rows=[['a,b', '1']],
                            lineterminator='\r\n', csv_sha256=hashlib.sha256(body).hexdigest())
            path.with_suffix('.json').write_text(json.dumps(snapshot))
            materialize_csv(path)
            self.assertEqual(path.read_bytes(), body)
            path.write_bytes(b'existing')
            materialize_csv(path)
            self.assertEqual(path.read_bytes(), b'existing')

    def test_reject_corrupt_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            path.with_suffix('.json').write_text(json.dumps(dict(
                columns=['path'], rows=[['a']], lineterminator='\n', csv_sha256='wrong')))
            with self.assertRaises(ValueError):
                materialize_csv(path)
            self.assertFalse(path.exists())
