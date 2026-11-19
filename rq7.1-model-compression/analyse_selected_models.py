"""Export/analyse only artifacts admitted to final_comparison.csv."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if cfg.get('max_eval_batches') is not None:raise ValueError('Full-data analysis requires unrestricted evaluation')
    root=Path(cfg['output_dir'])
    with (root/'final_comparison.csv').open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    destination=root/'prediction_analysis'
    destination.mkdir(exist_ok=False)
    inventory=[]
    for index,row in enumerate(rows):
        if row.get('valid_result','True').lower() not in ('true','1','1.0'):continue
        artifact=Path(row['artifact'])
        if not artifact.is_file():raise FileNotFoundError(artifact)
        key=f"{index:02d}_{row['variant']}_{row['model']}"
        predictions=destination/f'{key}.json'
        for script,arguments in [('export_predictions.py',['--config',str(args.config.resolve()),'--model',str(artifact.resolve()),'--output',str(predictions.resolve())]),('plot_prediction_analysis.py',['--input',str(predictions.resolve()),'--output',str((destination/key).resolve())])]:
            subprocess.run([sys.executable,'-u',str(Path(__file__).parent/script),*arguments],check=True)
        inventory.append(dict(key=key,seed=row['seed'],variant=row['variant'],model=row['model'],teacher=row.get('teacher_model'),artifact=str(artifact),prediction_file=str(predictions)))
    (destination/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')


if __name__=='__main__':main()
