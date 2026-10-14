"""Fixed classical segmentation and explicitly nonclinical feature rules."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from skimage.measure import regionprops, label, perimeter_crofton
from scipy.ndimage import binary_fill_holes

VERSION = '1.0.0'
MAX_SIDE = 512

@dataclass
class Features:
    asymmetry: float
    border_compactness: float
    colors: list
    color_fractions: dict
    relative_diameter: float
    area_fraction: float
    components: dict
    score: float
    band: str
    warnings: list
    def to_dict(self): return asdict(self)

def resize_rgb(rgb):
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<32: raise ValueError('Use an RGB image at least 32 x 32 pixels.')
    if rgb.dtype!=np.uint8: raise ValueError('RGB input must be uint8.')
    h,w=rgb.shape[:2]; scale=min(1,MAX_SIDE/max(h,w))
    return cv2.resize(rgb,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA) if scale<1 else rgb.copy()

def preprocess(rgb, clahe=False):
    rgb=resize_rgb(rgb); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    response=np.zeros_like(gray)
    # Dark, thin structures in several directions; broad lesion interiors are not targeted.
    for angle in [0,45,90,135]:
        k=np.zeros((17,17),np.uint8); cv2.line(k,(0,8),(16,8),1,1)
        k=cv2.warpAffine(k,cv2.getRotationMatrix2D((8,8),angle,1),(17,17),flags=cv2.INTER_NEAREST)
        response=np.maximum(response,cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,k))
    hair=((response>18)&(gray<160)).astype(np.uint8)*255
    hair=cv2.dilate(hair,np.ones((3,3),np.uint8))
    clean=cv2.inpaint(rgb,hair,3,cv2.INPAINT_TELEA)
    if clahe:
        lab=cv2.cvtColor(clean,cv2.COLOR_RGB2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0])
        clean=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    return rgb,clean,hair

def segment(clean):
    h,w=clean.shape[:2]
    lum=cv2.cvtColor(clean,cv2.COLOR_RGB2LAB)[:,:,0]
    lum=cv2.GaussianBlur(lum,(7,7),0)
    # Exclude dark dermatoscope corners; use interior for Otsu threshold estimation.
    yy,xx=np.mgrid[:h,:w]; ellipse=((xx-w/2)/(w*.49))**2+((yy-h/2)/(h*.49))**2<1
    valid=ellipse & (lum>15)
    values=lum[valid]
    if values.size==0: return np.zeros((h,w),bool)
    t,_=cv2.threshold(values.reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    candidate=((lum<t)&valid).astype(np.uint8)
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(candidate)
    if n<2: return np.zeros((h,w),bool)
    # Largest connected region, not the ground truth or a learned selector.
    largest=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    return binary_fill_holes(labels==largest)

def asymmetry(mask):
    points=np.argwhere(mask).astype(np.float64)
    if len(points)<10: return 0.0
    centered=points-points.mean(axis=0)
    _,vectors=np.linalg.eigh(np.cov(centered.T))
    aligned=centered@vectors
    # PCA-aligned centroid raster; compare each perpendicular reflection by XOR/union.
    aligned=np.rint(aligned).astype(int); margin=np.abs(aligned).max(axis=0)+2
    shape=tuple(2*margin+1); raster=np.zeros(shape,np.uint8)
    coords=aligned+margin; raster[coords[:,0],coords[:,1]]=1
    raster=cv2.morphologyEx(raster,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    scores=[]
    for axis in (0,1):
        flip=np.flip(raster,axis=axis); scores.append(np.logical_xor(raster,flip).sum()/max(1,np.logical_or(raster,flip).sum()))
    return float(np.mean(scores))

def features(rgb,mask):
    area=int(mask.sum()); h,w=mask.shape; warns=[]
    if area<max(50,.005*h*w): warns.append('Very small or missing region: the segmentation may be wrong.')
    if area>.75*h*w: warns.append('Most of the image was selected: the segmentation may be wrong.')
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any(): warns.append('The selected region touches an image edge and may be incomplete.')
    contrast=float(np.std(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)))
    if contrast<8: warns.append('Low contrast image: features are not reliable.')
    compact=max(1.0,float(perimeter_crofton(mask,4)**2/(4*np.pi*area))) if area else 1.0
    regions=regionprops(label(mask)); diameter=float(regions[0].axis_major_length/max(h,w)) if regions else 0.0
    a=asymmetry(mask)
    # Mutually exclusive, hand-set RGB/HSV bins. Fractions count only bins >=5% of region.
    px=rgb[mask].astype(float)/255 if area else np.zeros((0,3)); fractions={}
    if area:
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[mask].astype(float); hue=hsv[:,0]*2; sat=hsv[:,1]/255; val=hsv[:,2]/255
        r,g,b=px.T; remaining=np.ones(area,bool)
        rules=[('black',val<.20),('white',(val>.80)&(sat<.18)),
          ('red',((hue<18)|(hue>342))&(sat>.35)&(r>g*1.20)&(r>b*1.20)),
          ('blue-gray',(b>r*1.05)&(val<.80)),
          ('dark brown',(val<.52)&(r>=b)),('light brown',(r>=b)&(val>=.52)&(sat>=.15))]
        for name,condition in rules:
            chosen=remaining&condition; fractions[name]=float(chosen.sum()/area); remaining &= ~chosen
        fractions['unclassified']=float(remaining.sum()/area)
    else: fractions={x:0.0 for x in ['black','white','red','blue-gray','dark brown','light brown','unclassified']}
    colors=[n for n,f in fractions.items() if n!='unclassified' and f>=.05]
    # Diameter is shown but excluded: framing is not physical lesion size.
    parts={'asymmetry':float(np.clip(a/.45,0,1)), 'border':float(np.clip((compact-1)/1.5,0,1)),
           'color':float(np.clip((len(colors)-1)/4,0,1))}
    score=100*(.40*parts['asymmetry']+.30*parts['border']+.30*parts['color'])
    band='Low' if score<35 else 'Moderate' if score<65 else 'High'
    if warns: band='Unreliable'
    return Features(a,compact,colors,fractions,diameter,area/(h*w),parts,score,band,warns)

def analyze(rgb,clahe=False):
    original,clean,hair=preprocess(rgb,clahe); mask=segment(clean)
    # Colors measured on hair-cleaned original (not CLAHE, which changes colors).
    color_image=preprocess(original,False)[1] if clahe else clean
    f=features(color_image,mask)
    overlay=original.copy(); overlay[mask]=(original[mask]*.65+np.array([50,170,150])*.35).astype(np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay,contours,-1,(30,140,120),2)
    return {'original':original,'preprocessed':clean,'hair_mask':hair,'mask':mask,'overlay':overlay,'features':f}

def overlap(pred,truth):
    pred=np.asarray(pred,bool); truth=np.asarray(truth,bool)
    if pred.shape!=truth.shape: raise ValueError('Masks must have equal shape.')
    inter=np.logical_and(pred,truth).sum(); total=pred.sum()+truth.sum(); union=np.logical_or(pred,truth).sum()
    return {'dice':float(2*inter/total) if total else 1.0,'iou':float(inter/union) if union else 1.0}
