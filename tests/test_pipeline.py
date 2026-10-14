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
