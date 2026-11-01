"""Generate calibration and ID risk figures from stable saved logits."""
import argparse
import json
from pathlib import Path
from evaluate_saved_logits import analyse, scores
from selective_risk import risk_coverage


def render(payload, output):
    from matplotlib import pyplot as plt
    result=analyse(payload)
    output.mkdir(parents=True,exist_ok=False)
    risks={}
    for key in ('test_uncalibrated','test_calibrated'):
        metrics=result[key]
        temperature=metrics['temperature']
        confidence,errors=[],[]
        for row in payload['test']:
            p,_=scores(row['logits'],temperature)
            prediction=max(range(len(p)),key=p.__getitem__)
            confidence.append(p[prediction]);errors.append(int(prediction!=row['label']))
        risks[key]=risk_coverage(confidence,errors)
    result['id_selective_risk']=risks
    (output/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for kind in ('reliability','risk_coverage'):
        fig,ax=plt.subplots(figsize=(7,5))
        for key,label in [('test_uncalibrated','Uncalibrated'),('test_calibrated','Validation-selected temperature')]:
            if kind=='reliability':
                bins=[b for b in result[key]['reliability'] if b['count']]
                ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=label)
            else:
                points=risks[key]['points']
                ax.step([p['coverage'] for p in points],[p['risk'] for p in points],where='pre',label=label)
        if kind=='reliability':
            ax.plot([0,1],[0,1],'k--',label='Perfect calibration')
            ax.set(xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1))
        else:ax.set(xlabel='Accepted ID coverage',ylabel='Classification error among accepted images')
        ax.set_title('Single model/seed: '+kind.replace('_',' ')+' (ID only)')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(output/f'{kind}.{ext}',dpi=180)
        plt.close(fig)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    render(json.loads(args.input.read_text()),args.output)
