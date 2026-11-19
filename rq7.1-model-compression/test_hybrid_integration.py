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

MANIFESTS = Path(__file__).parent/'splits'

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
            original.parent.mkdir(parents=True)
            original.write_bytes(body)
            archive = root/'images.tar'
            with tarfile.open(archive, 'w') as handle:
                entry = tarfile.TarInfo('rq71_plantvillage/'+row['source_path'])
                entry.size = len(body)
                handle.addfile(entry, io.BytesIO(body))
            for i, input_path in enumerate((source, archive)):
                destination = root/f'output{i}'
                argv = ['stage', '--source', str(input_path), '--output', str(destination)]
                with patch.object(sys, 'argv', argv), patch.object(stage_plantvillage, 'validate', return_value=({}, [row])):
                    stage_plantvillage.main()
                    self.assertEqual((destination/row['path']).read_bytes(), body)
                    with self.assertRaises(FileExistsError):
                        stage_plantvillage.main()
            self.assertEqual(original.read_bytes(), body)

    def test_unsafe_paths_rejected(self):
        for value in ('../escape', '/absolute', 'C:/absolute', 'train\\escape'):
            with self.assertRaises(ValueError):
                hybrid_protocol.safe_relative(value)

if __name__ == '__main__':
    unittest.main()
