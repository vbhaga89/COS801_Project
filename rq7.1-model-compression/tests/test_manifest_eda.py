import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys
import tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))


class ManifestEdaTests(unittest.TestCase):
    def test_manifest_tuple_and_unlisted_copy(self):
        from rq71.manifests import prepare_manifests
        from rq71.data import audit_splits
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'images';splits=Path(folder)/'splits'
            for index,split in enumerate(('train','val','test')):
                dest=root/split/'Crop__healthy';dest.mkdir(parents=True)
                (dest/'image.jpg').write_bytes(f'unique image {index}'.encode())
            prepare_manifests(root,splits)
            # An excluded physical duplicate must not re-enter the manifest audit.
            (root/'test/Crop__healthy/excluded.jpg').write_bytes((root/'train/Crop__healthy/image.jpg').read_bytes())
            report=audit_splits(root,splits)
            self.assertEqual(report['totals'],dict(train=1,val=1,test=1))
            self.assertEqual(report['cross_split_exact_duplicate_count'],0)
            self.assertEqual(audit_splits(root)['cross_split_exact_duplicate_count'],1)


if __name__=='__main__':unittest.main()
