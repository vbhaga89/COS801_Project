import hashlib
import io
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hybrid_protocol
import stage_plantvillage

MANIFESTS = Path(__file__).parent/'splits_hybrid_v2'

class IntegrationTests(unittest.TestCase):
    def test_reviewed_protocol(self):
        protocol, rows = hybrid_protocol.validate(MANIFESTS)
        self.assertEqual(len(rows), 54299)
        self.assertEqual(protocol['counts'], hybrid_protocol.COUNTS)

    def test_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'manifests'
            shutil.copytree(MANIFESTS, folder)
            with (folder/'assignments.csv').open('ab') as handle:
                handle.write(b'\n')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                hybrid_protocol.validate(folder)

    def test_directory_and_archive_staging(self):
        body = b'fixture-image-bytes'
        row = dict(source_path='train/class/image.jpg', path='val/class/image.jpg', sha256=hashlib.sha256(body).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            original = source/row['source_path']
