"""Validate the ABCD-inspired score as a melanoma-vs-rest discriminator (ISIC 2018 Task 3 / HAM10000).

This measures whether the hand-set score ranks melanoma above other lesions in a labelled
sample. It is NOT clinical accuracy: no sensitivity/specificity at a clinical cutoff, no
patient-level split, dermoscopy images only, and the score was never tuned on these labels.
"""
import argparse, csv, io, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from remotezip import RemoteZip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skinlens.pipeline import analyze, VERSION
ROOT=Path(__file__).resolve().parents[1]
BASE='https://isic-archive.s3.amazonaws.com/challenges/2018/'
IMG_URL=BASE+'ISIC2018_Task3_Training_Input.zip'
GT_URL=BASE+'ISIC2018_Task3_Training_GroundTruth.zip'
GT_NAME='ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

def auc(score,y):
    """Rank-based (Mann-Whitney) AUC with average ranks for ties."""
    score=np.asarray(score,float); y=np.asarray(y,bool)
    order=np.argsort(score,kind='mergesort'); s=score[order]; ranks=np.empty(len(s))
    i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s[j+1]==s[i]: j+=1
        ranks[i:j+1]=(i+j)/2+1; i=j+1
    r=np.empty(len(s)); r[order]=ranks
    n1=y.sum(); n0=(~y).sum()
    return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))

def roc_points(score,y):
    thr=np.unique(score)[::-1]; y=np.asarray(y,bool)
    tpr=[0.0]+[float((score[y]>=t).mean()) for t in thr]; fpr=[0.0]+[float((score[~y]>=t).mean()) for t in thr]
    return fpr,tpr

def boot_ci(score,y,n=2000,seed=2026):
    rng=np.random.default_rng(seed); pos=np.flatnonzero(y); neg=np.flatnonzero(~y); out=[]
    for _ in range(n):  # stratified resampling keeps both classes present
        idx=np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))])
        out.append(auc(score[idx],y[idx]))
    return np.quantile(out,[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--melanoma',type=int,default=150);p.add_argument('--other',type=int,default=350)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--data',type=Path,default=ROOT/'data'/'task3')
    p.add_argument('--output',type=Path,default=ROOT/'reports'/'validation');a=p.parse_args()
    a.data.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    with RemoteZip(GT_URL,timeout=90) as z: rows=list(csv.DictReader(io.StringIO(z.read(GT_NAME).decode())))
    mel=sorted(r['image'] for r in rows if float(r['MEL'])==1); other=sorted(r['image'] for r in rows if float(r['MEL'])!=1)
    rng=np.random.default_rng(a.seed)
    ids=[(i,1) for i in rng.choice(mel,a.melanoma,replace=False)]+[(i,0) for i in rng.choice(other,a.other,replace=False)]
    dx={r['image']:[k for k in ['MEL','NV','BCC','AKIEC','BKL','DF','VASC'] if float(r[k])==1][0] for r in rows}
    with RemoteZip(IMG_URL,timeout=90) as z:
        names={Path(n).stem:n for n in z.namelist() if n.endswith('.jpg')}
        for ident,_ in ids:
            dest=a.data/f'{ident}.jpg'
            if not dest.exists():
                for t in range(3):
                    try: dest.write_bytes(z.read(names[ident]));break
                    except Exception:
                        if t==2: raise
                        time.sleep(2)
    out=[]
    for k,(ident,label) in enumerate(ids):
        res=analyze(np.array(Image.open(a.data/f'{ident}.jpg').convert('RGB')));f=res['features']
        out.append({'id':ident,'dx':dx[ident],'melanoma':label,'score':f.score,'band':f.band,'flagged':bool(f.warnings),
                    'asymmetry_part':f.components['asymmetry'],'border_part':f.components['border'],'color_part':f.components['color']})
        if (k+1)%50==0: print(f'{k+1}/{len(ids)} analyzed',flush=True)
    with (a.output/'score_validation.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
    y=np.array([r['melanoma'] for r in out],bool);s=np.array([r['score'] for r in out]);fl=np.array([r['flagged'] for r in out])
    def block(mask):
        return {'n':int(mask.sum()),'melanoma':int(y[mask].sum()),'auc':auc(s[mask],y[mask]),'auc_bootstrap_95ci':boot_ci(s[mask],y[mask])}
    comps={c:{'auc':auc(np.array([r[c+'_part'] for r in out]),y)} for c in ['asymmetry','border','color']}
    summary={'pipeline_version':VERSION,'dataset':'ISIC 2018 Task 3 training (HAM10000) ground truth, melanoma (MEL) vs all other classes',
      'sampling':f'seed {a.seed}; {a.melanoma} melanoma + {a.other} non-melanoma drawn without replacement (melanoma enriched vs ~11% in the dataset; AUC does not depend on prevalence)',
      'all_images':block(np.ones(len(out),bool)),'unflagged_only':block(~fl),'flagged_count':int(fl.sum()),
      'flagged_melanoma_fraction':float(y[fl].mean()) if fl.any() else None,
      'component_auc':comps,'bands':{b:{'n':sum(r['band']==b for r in out),'melanoma':sum(r['band']==b and r['melanoma'] for r in out)} for b in ['Low','Moderate','High','Unreliable']},
      'bootstrap':'2000 stratified resamples, seed 2026','chance_auc':0.5,
      'scope':'Measures how well the hand-set score ranks melanoma above other lesions on dermoscopy images. Not clinical accuracy, sensitivity/specificity, or patient-level performance.'}
    (a.output/'score_validation.json').write_text(json.dumps(summary,indent=2))
    fpr,tpr=roc_points(s,y);A=summary['all_images']
    fig,ax=plt.subplots(figsize=(5.5,5.5),layout='constrained')
    ax.plot(fpr,tpr,color='#356d5f',lw=2.2,label=f'SkinLens score (AUC {A["auc"]:.3f}, 95% CI {A["auc_bootstrap_95ci"][0]:.3f}-{A["auc_bootstrap_95ci"][1]:.3f})')
    ax.plot([0,1],[0,1],'--',color='#999',label='Chance (AUC 0.5)')
    ax.set(xlabel='False positive rate (non-melanoma scored above cutoff)',ylabel='True positive rate (melanoma scored above cutoff)',xlim=(0,1),ylim=(0,1))
    ax.set_title(f'Melanoma vs rest | HAM10000 sample, n={len(out)}',loc='left',fontsize=12);ax.legend(loc='lower right',frameon=False,fontsize=8)
    fig.text(.02,-.02,'ISIC 2018 Task 3 | CC-BY-NC | Hand-set score, not tuned on these labels. Not clinical accuracy.',fontsize=7,color='#585254')
    fig.savefig(a.output/'roc_melanoma.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
