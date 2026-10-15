import numpy as np
import cv2
import pytest
from skinlens.pipeline import analyze, overlap, features, asymmetry

def synthetic():
    im=np.full((256,256,3),(215,175,150),np.uint8)
    cv2.circle(im,(128,128),55,(85,45,30),-1)
    return im

def test_circle():
    res=analyze(synthetic()); yy,xx=np.mgrid[:256,:256];gt=(xx-128)**2+(yy-128)**2<=55**2
    assert overlap(res['mask'],gt)['dice']>.93
    assert 0<=res['features'].score<=100
    assert res['features'].asymmetry<.12
    assert 1<=res['features'].border_compactness<1.2

def test_overlap():
    a=np.zeros((10,10),bool);assert overlap(a,a)=={'dice':1.0,'iou':1.0}
    b=~a;assert overlap(a,b)=={'dice':0.0,'iou':0.0}
    with pytest.raises(ValueError):overlap(a,np.zeros((2,2)))

def test_blank_withheld():
    res=analyze(np.full((100,100,3),150,np.uint8))
    assert res['features'].band=='Unreliable'

def test_scale_and_determinism():
    im=cv2.resize(synthetic(),(1200,900));a=analyze(im);b=analyze(im)
    assert a['mask'].shape==(384,512)
    assert np.array_equal(a['mask'],b['mask'])
    assert a['features'].to_dict()==b['features'].to_dict()

def test_clahe_and_colors():
    r=analyze(synthetic(),True)
    assert len(r['features'].colors)>=1
    assert sum(r['features'].color_fractions.values())==pytest.approx(1)

def test_bad_input():
    with pytest.raises(ValueError):analyze(np.zeros((8,8,3),np.uint8))
    with pytest.raises(ValueError):analyze(np.zeros((100,100),np.uint8))


def _load_script(name):
    import importlib.util, pathlib
    path=pathlib.Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_auc_known_values():
    v=_load_script('validate_score')
    assert v.auc([1,2,3,4],[0,0,1,1])==1.0
    assert v.auc([4,3,2,1],[0,0,1,1])==0.0
    assert v.auc([1,1,1,1],[0,1,0,1])==0.5

def test_baseline_variants_return_masks():
    ev=_load_script('evaluate'); im=synthetic()
    masks={v:ev.analyze_variant(im,v)['mask'] for v in ev.VARIANTS}
    assert all(m.shape==im.shape[:2] for m in masks.values())
    assert masks['otsu_gray'].sum()>0 and masks['otsu_gray'].dtype==bool

def test_bundled_samples_show_band_variety():
    import json, pathlib
    from PIL import Image
    root=pathlib.Path(__file__).resolve().parents[1]/'samples'
    bands={p.stem:analyze(np.array(Image.open(p).convert('RGB')))['features'].band for p in root.glob('*.jpg')}
    assert {'Moderate','High','Unreliable'} <= set(bands.values()), bands
    assert json.loads((root/'credits.json').read_text())['license']=='CC-BY-NC 4.0'
