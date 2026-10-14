"""Fixed, image-wise segmentation evaluation; no cancer classification claim."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skinlens.pipeline import analyze, overlap, VERSION
ROOT=Path(__file__).resolve().parents[1]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#251f21','axes.titlecolor':'#251f21',
                     'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})

def result_figure(ident,image,truth,result,metrics,path,label_text):
    fig,axs=plt.subplots(1,4,figsize=(14,4.5),layout='constrained')
    for ax,im,title in zip(axs,[result['original'],result['preprocessed'],truth,result['overlay']],
                           ['Original dermoscopy','Hair removal','Expert reference mask','Classical segmentation']):
        ax.imshow(im,cmap='gray' if im.ndim==2 else None,vmin=0,vmax=1 if im.ndim==2 else None)
        ax.set_title(title,fontsize=12,fontweight='bold',pad=10); ax.axis('off')
    fig.suptitle(f'SkinLens | {label_text}\n{ident}  -  Dice {metrics["dice"]:.3f}  /  IoU {metrics["iou"]:.3f}',fontsize=17,x=.02,ha='left')
    fig.text(.02,.015,'ISIC 2018 Task 1 | CC-BY-NC | Teal = predicted lesion. Segmentation overlap is not melanoma accuracy.',fontsize=9,color='#585254')
    fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.2);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'reports')
    p.add_argument('--split',choices=['development','evaluation'],default='evaluation');p.add_argument('--clahe',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.data/'manifest.json').read_text());records=[r for r in manifest['records'] if r['split']==a.split]
    rows=[]
    for i,record in enumerate(records):
        ident=record['id'];impath=a.data/'images'/f'{ident}.jpg';mpath=a.data/'masks'/f'{ident}.png'
        for path,key in [(impath,'image_sha256'),(mpath,'mask_sha256')]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record[key]:raise ValueError(f'Checksum mismatch: {path}')
        image=np.array(Image.open(impath).convert('RGB'));truth=np.array(Image.open(mpath).convert('L'))>127
        if truth.shape!=image.shape[:2]:raise ValueError(f'Original mask/image dimensions disagree: {ident}')
        result=analyze(image,a.clahe);truth=cv2.resize(truth.astype(np.uint8),result['mask'].shape[::-1],interpolation=cv2.INTER_NEAREST).astype(bool)
        metrics=overlap(result['mask'],truth);rows.append({'id':ident,**metrics,'feature_score':result['features'].score,'feature_band':result['features'].band,
            'quality_flagged':bool(result['features'].warnings)})
        if (i+1)%25==0: print(f'{i+1}/{len(records)} evaluated',flush=True)
    if not rows:raise ValueError('No records in requested split')
    values={k:np.array([r[k] for r in rows]) for k in ['dice','iou']}
    rng=np.random.default_rng(2026);means={k:np.mean(v[rng.integers(0,len(v),(2000,len(v)))],axis=1) for k,v in values.items()}
    summary={'pipeline_version':VERSION,'split':a.split,'n':len(rows),'dataset':manifest['dataset'],'subset_seed':manifest['seed'],
      'max_side':512,'clahe':a.clahe,'metric_space':'Masks resized with nearest-neighbor to aspect-preserving max-side 512 images.',
      'mean_dice':float(values['dice'].mean()),'median_dice':float(np.median(values['dice'])),
      'mean_iou':float(values['iou'].mean()),'median_iou':float(np.median(values['iou'])),
      'dice_std':float(values['dice'].std()),'iou_std':float(values['iou'].std()),
      'mean_dice_bootstrap_95ci':np.quantile(means['dice'],[.025,.975]).tolist(),
      'mean_iou_bootstrap_95ci':np.quantile(means['iou'],[.025,.975]).tolist(),
      'iou_below_05':int((values['iou']<.5).sum()),'quality_flagged':sum(r['quality_flagged'] for r in rows),
      'clinical_classification_evaluated':False,'code_sha256':hashlib.sha256((ROOT/'skinlens'/'pipeline.py').read_bytes()).hexdigest()}
    (a.output/'metrics.json').write_text(json.dumps(summary,indent=2))
    with (a.output/'per_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    ordered=sorted(rows,key=lambda r:r['iou'])
    chosen=[(ordered[round(.9*(len(rows)-1))],'Strong overlap (90th percentile)','example_strong.png'),
            (ordered[(len(rows)-1)//2],'Typical overlap (median)','example_typical.png'),
            (ordered[0],'Failure case (lowest overlap)','example_failure.png')]
    for row,caption,filename in chosen:
        ident=row['id'];image=np.array(Image.open(a.data/'images'/f'{ident}.jpg').convert('RGB'));res=analyze(image,a.clahe)
        truth=cv2.resize(np.array(Image.open(a.data/'masks'/f'{ident}.png').convert('L')),res['mask'].shape[::-1],interpolation=cv2.INTER_NEAREST)>127
        result_figure(ident,image,truth,res,row,a.output/filename,caption)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.hist(values['iou'],bins=np.linspace(0,1,21),color='#73a89a',edgecolor='white');ax.set(xlabel='Intersection over union (IoU)',ylabel='Number of images',xlim=(0,1))
    ax.set_title(f'Classical segmentation | {len(rows)} {a.split} images',loc='left',fontsize=16,pad=16)
    ax.axvline(values['iou'].mean(),color='#251f21',linestyle='--',label=f'Mean IoU: {values["iou"].mean():.3f}');ax.legend(frameon=False)
    fig.savefig(a.output/'overlap_distribution.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
