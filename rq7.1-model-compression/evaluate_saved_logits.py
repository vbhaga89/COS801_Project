"""Prospective post-hoc evaluation. No torch, inference, training or GPU imports.

Input JSON: class_names, class_crops, validation/test record lists (id,label,logits),
and provenance. Calibration is selected on validation NLL; test is never tuned on.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

def validate(payload):
    names, crops = payload['class_names'], payload['class_crops']
    if len(names) < 2 or len(set(names)) != len(names) or len(crops) != len(names):
        raise ValueError('Need unique class names and one explicit crop name per class.')
    if not all(isinstance(v,str) and v for v in names+crops):
        raise ValueError('Class and crop names must be nonempty strings.')
    sets = []
    for split in ('validation','test'):
        rows = payload[split]
        if not rows:
            raise ValueError(f'{split} records are empty.')
        ids = set()
        for row in rows:
            ident, label, logits = row['id'], row['label'], row['logits']
            if not isinstance(ident,str) or not ident or ident in ids:
                raise ValueError(f'{split} has missing/duplicate sample IDs.')
            ids.add(ident)
            if type(label) is not int or not 0 <= label < len(names):
                raise ValueError('Label outside declared class mapping.')
            if len(logits) != len(names) or not all(type(v) in (int,float) and math.isfinite(v) for v in logits):
                raise ValueError('Logits must be finite, numeric, and match class count.')
        sets.append(ids)
    if sets[0] & sets[1]:
        raise ValueError('Validation/test sample IDs overlap.')
    if not isinstance(payload.get('provenance'),dict) or not payload['provenance']:
        raise ValueError('Provide model/seed/manifest provenance; it is retained, not independently certified.')

def scores(logits, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be positive and finite.')
    scaled = [float(v)/temperature for v in logits]
    peak = max(scaled)
    log_sum = peak + math.log(sum(math.exp(v-peak) for v in scaled))
    return [math.exp(v-log_sum) for v in scaled], [v-log_sum for v in scaled]

def nll(rows, temperature):
    return sum(-scores(r['logits'],temperature)[1][r['label']] for r in rows)/len(rows)

def choose_temperature(validation):
    curve = [{'temperature':t,'validation_nll':nll(validation,t)} for t in TEMPERATURES]
    # Ties favour no change, then the smaller distance from T=1.
    best = min(curve,key=lambda r:(r['validation_nll'],abs(r['temperature']-1),r['temperature']))
    return best['temperature'], curve

def evaluate(rows, names, crops, temperature=1.0, bins=15):
    if type(bins) is not int or bins <= 0:
        raise ValueError('bins must be a positive integer.')
    count = len(rows)
    matrix = [[0]*len(names) for _ in names]
    buckets = [[] for _ in range(bins)]
    brier, errors = 0.0, []
    for row in rows:
        p,_ = scores(row['logits'],temperature)
        pred = max(range(len(p)),key=p.__getitem__)
        true = row['label']
        matrix[true][pred] += 1
        confidence, correct = p[pred], pred == true
        buckets[min(bins-1,int(confidence*bins))].append((confidence,int(correct)))
        brier += sum((value-int(i==true))**2 for i,value in enumerate(p))
        if not correct:
            errors.append(dict(id=row['id'],true_class=names[true],predicted_class=names[pred],
                               true_crop=crops[true],confidence=confidence))
    classes=[]
    for i,name in enumerate(names):
        tp=matrix[i][i]; support=sum(matrix[i]); predicted=sum(row[i] for row in matrix)
        precision=tp/predicted if predicted else 0.0
        recall=tp/support if support else None
        f1=2*tp/(support+predicted) if support+predicted else 0.0
        classes.append(dict(name=name,crop=crops[i],support=support,precision=precision,recall=recall,f1=f1))
    crop_rows=[]
    for crop in sorted(set(crops)):
        indices=[i for i,c in enumerate(crops) if c==crop]
        support=sum(sum(matrix[i]) for i in indices)
        crop_rows.append(dict(crop=crop,support=support,
                              accuracy=sum(matrix[i][i] for i in indices)/support if support else None))
    reliability=[]; ece=0.0
    for i,bucket in enumerate(buckets):
        confidence=sum(x[0] for x in bucket)/len(bucket) if bucket else None
        accuracy=sum(x[1] for x in bucket)/len(bucket) if bucket else None
        if bucket:
            ece += len(bucket)/count * abs(confidence-accuracy)
        reliability.append(dict(lower=i/bins,upper=(i+1)/bins,count=len(bucket),
                                confidence=confidence,accuracy=accuracy))
    supported=[row for row in classes if row['support']]
    supported_crops=[row for row in crop_rows if row['support']]
    errors.sort(key=lambda r:r['confidence'],reverse=True)
    return dict(n=count,temperature=temperature,accuracy=sum(matrix[i][i] for i in range(len(names)))/count,
                macro_f1=sum(row['f1'] for row in classes)/len(names),
                nll=nll(rows,temperature),brier_multiclass_sum=brier/count,ece_top_label=ece,
                reliability=reliability,confusion_matrix=matrix,per_class=classes,per_crop=crop_rows,
                worst_supported_class_recall=min(supported,key=lambda r:r['recall']),
                worst_supported_crop_accuracy=min(supported_crops,key=lambda r:r['accuracy']),
                most_confident_errors=errors[:30])

def analyse(payload):
    validate(payload)
    temperature, curve=choose_temperature(payload['validation'])
    names,crops=payload['class_names'],payload['class_crops']
    before=evaluate(payload['test'],names,crops)
    after=evaluate(payload['test'],names,crops,temperature)
    assert before['confusion_matrix']==after['confusion_matrix'], 'Positive scaling must preserve predictions.'
    return dict(provenance=payload['provenance'],status='post-hoc evaluation, not retraining',
                selection_split='validation',selection_metric='NLL',temperature_grid=list(TEMPERATURES),
                validation_search=curve,test_uncalibrated=before,test_calibrated=after,
                limitations=['ECE uses 15 fixed equal-width top-label confidence bins.',
                             'Worst groups use supported classes/crops only; read their support counts.',
                             'Macro F1 uses all declared classes, assigning zero when F1 is undefined.',
                             'Group labels are explicit class metadata, not acquisition device/geography.',
                             'Disjoint IDs do not establish image-hash or leaf-group independence.',
                             'Training overlap, logits origin and checkpoint/manifest integrity require an external audit.',
                             'Use actual exported-model logits to evaluate compression calibration; do not substitute teacher logits.'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.input.read_bytes()
    result=analyse(json.loads(raw))
    result['input_sha256']=hashlib.sha256(raw).hexdigest()
    # Exclusive creation protects previous analyses from accidental overwrite.
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(args.output)

if __name__=='__main__':
    main()
