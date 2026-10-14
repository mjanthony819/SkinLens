"""Educational classical-DIP dermoscopy demo."""
import io, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
from skinlens.pipeline import analyze

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='SkinLens | Explainable lesion analysis',page_icon='🔬',layout='wide')
st.markdown('''<style>
.stApp {background: #faf9f7; color: #251f21;}
.block-container {max-width: 1150px; padding-top: 4rem;}
h1,h2,h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: white; padding: 18px; border: 1px solid #eae9ea; border-radius: 12px;}
[data-testid="stMetricValue"] {color: #356d5f; font-size: 1.65rem;}
[data-testid="stAlert"] {border-radius: 10px;}
</style>''',unsafe_allow_html=True)
st.caption('CLASSICAL IMAGE PROCESSING  /  CODETHECURE  /  DIP PORTFOLIO')
st.title('SkinLens')
st.write('See how a lesion is segmented. Understand every feature behind the score.')
st.warning('Screening aid, not a diagnosis - see a dermatologist. This is an educational prototype, not a medical device. A low score does not rule out melanoma. If a lesion is new, changing, bleeding, or concerning, seek medical care regardless of this result.')

with st.sidebar:
    st.header('Analyze an image')
    upload=st.file_uploader('Upload a lesion image',type=['jpg','jpeg','png'],help='Best suited to close-up dermoscopic images. Phone photos are not validated.')
    sample_paths=sorted((ROOT/'samples').glob('*.jpg'))
    choice=st.radio('Or open a bundled sample',[p.stem for p in sample_paths],index=0) if sample_paths else None
    clahe=st.checkbox('Enhance contrast with CLAHE',False,help='Optional. Default evaluation uses CLAHE off. Color features use non-enhanced pixels.')
    st.caption('When hosted online, uploaded images are sent to the hosting server for processing. This demo does not deliberately save uploads, but hosting infrastructure may retain logs or temporary data. Use bundled samples; do not upload patient, identifiable, or sensitive images. When run locally, processing stays on your machine.')
    st.caption('Samples: ISIC 2018 Task 1, CC-BY-NC. See DATA_LICENSE.md for credits.')

source=upload if upload is not None else (ROOT/'samples'/f'{choice}.jpg' if choice else None)
if source is None:
    st.info('Upload a dermoscopic image, or run the download script to obtain sample data.');st.stop()
try:
    if upload is not None and upload.size>15*1024*1024: raise ValueError('Please use an image smaller than 15 MB.')
    with Image.open(source) as im:
        if im.width*im.height>20_000_000: raise ValueError('Please resize to at most 20 megapixels before upload.')
        if min(im.size)<32: raise ValueError('Use an image at least 32 x 32 pixels.')
        rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'))
    with st.spinner('Removing hair and extracting the lesion...'): result=analyze(rgb,clahe)
except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError) as e:
    st.error(f'Could not analyze this image: {e}');st.stop()
f=result['features']
st.subheader('From image to lesion')
cols=st.columns(3)
for col,key,title in zip(cols,['original','preprocessed','overlay'],['01  Original','02  Hair removal + preprocessing','03  Segmentation overlay']):
    with col: st.image(result[key],caption=title,width='stretch')
st.caption('Teal marks the predicted lesion. Always check this outline first: an incorrect mask makes the feature score unreliable.')

st.subheader('Explainable feature score')
if f.warnings:
    st.error('Result withheld: segmentation or image quality checks failed. ' + ' '.join(f.warnings))
else:
    st.info(f'{f.band} feature-based risk band (demo only) | {f.score:.1f} / 100. This band summarizes visual irregularity, not the probability of cancer. No clinical accuracy has been established.')
cols=st.columns(4)
cols[0].metric('A | Shape asymmetry',f'{f.asymmetry:.3f}');cols[0].caption('0 = symmetric. PCA-aligned reflection mismatch.')
cols[1].metric('B | Border compactness',f'{f.border_compactness:.2f}');cols[1].caption('1 = circle. Higher values indicate a less regular outline.')
cols[2].metric('C | Color groups',str(len(f.colors)));cols[2].caption(', '.join(f.colors) if f.colors else 'No color bin reached 5% of lesion area.')
cols[3].metric('D | Relative diameter',f'{100*f.relative_diameter:.1f}%');cols[3].caption('Major axis / longest image side. Not millimeters; excluded from scoring.')
with st.expander('How the score works'):
    st.write('This is a custom ABCD-inspired rule, not the clinical total dermoscopy score (TDS). Clinical dermoscopy ABCD uses D for structures, not diameter.')
    st.code('a = clip(asymmetry / 0.45, 0, 1)\nb = clip((compactness - 1) / 1.5, 0, 1)\nc = clip((color_count - 1) / 4, 0, 1)\nscore = 100 * (0.40*a + 0.30*b + 0.30*c)\nLow: <35 | Moderate: 35 to <65 | High: >=65',language='python')
    st.write('Weights and thresholds are hand-set for demonstration. Diameter cannot be interpreted without a physical scale and is not used. No diagnosis labels or cancer classifier are used.')
    st.json({'normalized_components':f.components,'color_area_fractions':f.color_fractions,'segmented_area_fraction':f.area_fraction})
with st.expander('Masks and downloadable results'):
    st.image(result['mask'].astype(np.uint8)*255,caption='Binary segmentation mask',width=300)
    mask_buffer=io.BytesIO();Image.fromarray(result['mask'].astype(np.uint8)*255).save(mask_buffer,format='PNG')
    st.download_button('Download mask',mask_buffer.getvalue(),'skinlens_mask.png','image/png')
    st.download_button('Download feature report',json.dumps({'prototype':'SkinLens','clinical_use':False,'clahe':clahe,**f.to_dict()},indent=2),'skinlens_features.json','application/json')

metric_path=ROOT/'reports'/'metrics.json'
if metric_path.exists():
    m=json.loads(metric_path.read_text());st.divider();st.subheader('Measured segmentation performance')
    st.write(f'ISIC 2018 Task 1 | {m["n"]} evaluation images | Mean Dice {m["mean_dice"]:.3f} | Mean IoU {m["mean_iou"]:.3f}')
    st.caption('Measured mask overlap only, not sensitivity, specificity, melanoma accuracy, or clinical safety. Metrics apply to the default settings. Evaluation and failure examples are in reports/.')
st.caption('Classical only: morphology, inpainting, Otsu thresholding, connected components, geometric features. No CNNs, no deep learning, no hidden model.')
