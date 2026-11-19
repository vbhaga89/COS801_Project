"""Stage exactly the audited records to fresh runtime-local storage, with hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).parent/'src'))
from rq71.manifests import read_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--manifests',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();total=0
    for split in ('train','val','test'):
        samples,_=read_manifest(args.source,args.manifests,split)
        with (args.manifests/f'{split}.csv').open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
        print('Staging',split,len(rows),'images',flush=True)
        for index,((path,_),row) in enumerate(zip(samples,rows),1):
            content=Path(path).read_bytes()
            if hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError(f'Hash mismatch: {path}')
            destination=args.destination/row['path']
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(content)
            total+=1
            if index==1 or index%500==0:print(f'{split}: {index}/{len(rows)}; total {total}; elapsed {time.monotonic()-started:.1f}s',flush=True)
    (args.destination/'staging_verified.json').write_text(json.dumps(dict(images=total,elapsed_seconds=time.monotonic()-started),indent=2))
    print('STAGING COMPLETE',total,flush=True)


if __name__=='__main__':main()
