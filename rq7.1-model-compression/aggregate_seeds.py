"""Aggregate explicit seed directories; retain selection-policy labels and n."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarise(directories):
    groups={};seen=set();statuses=[];reference=None
    for directory in directories:
        manifest=json.loads((directory/'full_run_manifest.json').read_text())
        seed=int(manifest['seed'])
        config=json.loads((directory.parent/f'config_seed{seed}.json').read_text())
        protocol=json.loads((Path(config['manifest_dir'])/'split_protocol.json').read_text())
        excluded={'seed','output_dir','data_dir','manifest_dir','_config_path'}
        settings={k:v for k,v in config.items() if k not in excluded}
        signature=(settings,protocol['files_sha256'])
        if reference is None:reference=signature
        elif signature!=reference:raise ValueError('Scientific configuration or split-hash mismatch')
        if seed in seen:raise ValueError('Duplicate seed directory')
        seen.add(seed)
        if manifest['status'] not in ('complete','partial'):raise ValueError('Unfinished/failed run cannot be aggregated')
        statuses.append(dict(seed=seed,status=manifest['status'],incomplete_stages=manifest.get('incomplete_stages',[])))
        with (directory/'final_comparison.csv').open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
        keys=set()
        for row in rows:
            if int(row['seed'])!=seed:raise ValueError('Seed mismatch in metrics')
            key=(row['model'],row['variant'],row.get('teacher_model',''))
            if key in keys:raise ValueError('Ambiguous duplicate selected-result row')
            keys.add(key)
            groups.setdefault(key,[]).append(row)
    if seen!={14,27,42}:raise ValueError('Headline aggregation requires exactly seeds 14,27,42')
    summary=[]
    for key,rows in sorted(groups.items()):
        result=dict(model=key[0],variant=key[1],teacher=key[2],n=len(rows),seeds=[int(r['seed']) for r in rows],compression_by_seed={r['seed']:r['compression'] for r in rows})
        for metric in ('accuracy','macro_f1','accuracy_drop','model_size_mb','median_latency_ms'):
            values=[float(r[metric]) for r in rows if r.get(metric) not in ('',None)]
            result[metric]=dict(n=len(values),mean=statistics.mean(values) if values else None,sample_sd=statistics.stdev(values) if len(values)>1 else None)
        summary.append(result)
    return dict(run_mode='SMOKE_ONLY' if reference[0].get('smoke_mode') else 'official_candidate',seeds=sorted(seen),statuses=statuses,summary=summary,warning='Audit resolved configurations, split hashes and environments before reporting. Missing INT8 results are not imputed; inspect incomplete stages.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('seed_directories',type=Path,nargs=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarise(args.seed_directories)
    with args.output.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2,allow_nan=False)
