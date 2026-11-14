import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
a=argparse.ArgumentParser();a.add_argument('analysis_dir');args=a.parse_args();root=Path(args.analysis_dir)
status=json.loads((root/'reproducibility/analysis_metadata.json').read_text())['lab_split_status']
rows=[]
for path in sorted((root/'predictions').glob('*_field.npz')):
 s=path.name[:-len('_field.npz')];f=np.load(path);l=np.load(root/'predictions'/f'{s}_lab.npz')
 y,p=f['y_true'],f['y_pred'];ly,lp=l['y_true'],l['y_pred'];ids=np.unique(y);all_ids=np.arange(len(f['class_names']));m=np.isin(ly,ids)
 def f1(y,p,ids):return f1_score(y,p,labels=ids,average='macro',zero_division=0)*100
 rows.append(dict(strategy=s,lab_accuracy_pct=accuracy_score(ly,lp)*100,field_accuracy_pct=accuracy_score(y,p)*100,lab_macro_f1_38_pct=f1(ly,lp,all_ids),field_macro_f1_38_pct=f1(y,p,all_ids),all38_f1_gap_pp=f1(ly,lp,all_ids)-f1(y,p,all_ids),matched_lab_accuracy_pct=accuracy_score(ly[m],lp[m])*100,matched_accuracy_gap_pp=100*(accuracy_score(ly[m],lp[m])-accuracy_score(y,p)),matched_lab_f1_pct=f1(ly[m],lp[m],ids),field_macro_f1_28_pct=f1(y,p,ids),matched_f1_gap_pp=f1(ly[m],lp[m],ids)-f1(y,p,ids),lab_split_status=status))
if not rows:raise RuntimeError('No saved predictions found')
out=pd.DataFrame(rows);out.to_csv(root/'reports/main_comparison_reproduced.csv',index=False);print(out.to_string(index=False))
