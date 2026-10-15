"""Quality-gate analysis from reports/per_image.csv: how many segmentation failures does the gate catch?"""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'reports'/'per_image.csv'); d['fail']=d.iou<.5
def row(x): return {'n':int(len(x)),'mean_dice':float(x.dice.mean()),'median_dice':float(x.dice.median()),'mean_iou':float(x.iou.mean()),'median_iou':float(x.iou.median()),'iou_below_05':int(x.fail.sum())}
fl=d[d.quality_flagged]; un=d[~d.quality_flagged]
out={'all':row(d),'flagged':row(fl),'unflagged':row(un),
 'failures_total':int(d.fail.sum()),'failures_caught_by_gate':int(fl.fail.sum()),'failures_missed_by_gate':int(un.fail.sum()),
 'gate_recall_of_failures':float(fl.fail.sum()/d.fail.sum()),'unflagged_failure_rate':float(un.fail.mean()),
 'unreliable_band_rate':float((d.feature_band=='Unreliable').mean()),'unreliable_band_n':int((d.feature_band=='Unreliable').sum()),
 'band_counts':{k:int(v) for k,v in d.feature_band.value_counts().items()}}
(ROOT/'reports'/'quality_gate.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
