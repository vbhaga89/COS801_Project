"""Stage reviewed image assignments into a NEW directory; never modify the source."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from hybrid_protocol import validate, safe_relative, PROTOCOL_SHA

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original image root or .tar archive')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, default=Path(__file__).parent/'splits_hybrid_v2')
    parser.add_argument('--archive-prefix', default='rq71_plantvillage')
    args = parser.parse_args()
    _, rows = validate(args.manifest_dir)
    source = args.source.resolve(strict=True)
    target = args.output.resolve()
    if target == source or source in target.parents:
        raise ValueError('Output must be separate from source')
    safe_relative(args.archive_prefix)
    target.mkdir(parents=True, exist_ok=False)
    archive = None
    try:
        if not source.is_dir():
            archive = tarfile.open(source)
        for index, row in enumerate(rows, 1):
            if archive:
                member = archive.getmember(args.archive_prefix.rstrip('/')+'/'+row['source_path'])
                if not member.isfile():
                    raise ValueError('Archive image is not a regular file')
                with archive.extractfile(member) as handle:
                    body = handle.read()
            else:
                image = (source/row['source_path']).resolve(strict=True)
                if source not in image.parents:
                    raise ValueError('Image escapes source directory')
                body = image.read_bytes()
            if hashlib.sha256(body).hexdigest() != row['sha256']:
                raise ValueError(f'Image hash mismatch: {row["source_path"]}')
            destination = target/row['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(body)
            if index % 5000 == 0:
                print(f'Staged and verified {index}/{len(rows)}', flush=True)
        (target/'STAGING_COMPLETE.json').write_text(json.dumps(dict(images=len(rows), protocol_sha256=PROTOCOL_SHA), indent=2))
        print('Staging complete:', target, flush=True)
    finally:
        if archive:
            archive.close()

if __name__ == '__main__':
    main()
