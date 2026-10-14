# Three-minute walkthrough

## 0:00-0:25 | Problem and scope
"SkinLens is an educational lesion-analysis prototype. We use classical digital
image processing to make every step visible. This is a screening aid, not a
diagnosis. It cannot tell whether a lesion is melanoma."

## 0:25-1:15 | Live app
Run `bash run.sh`. Choose a bundled dermoscopy sample. Point to the original,
hair-cleaned image, and teal predicted boundary. Switch sample images. Show the
CLAHE toggle and explain that the default benchmark leaves it off.

## 1:15-1:55 | Explainability
Point to A, B, C, D. Explain PCA reflection mismatch, perimeter/area compactness,
and simple color bins. Open "How the score works". The score is hand-set and
nonclinical; the displayed relative diameter is excluded because there is no
millimeter scale. Clinical TDS is a different rule and is not implemented here.

## 1:55-2:35 | Evidence, including failure
Show the mean Dice 0.662 and mean IoU 0.555 on 250 held-out image-level samples
from the ISIC 2018 training release. Open `reports/example_failure.png`: the
largest-component rule selected an artifact rather than the small faint lesion.
84 of 250 images have IoU below 0.5. A good-looking demo does not prove safety.

## 2:35-3:00 | Reproducibility and next steps
Show the source pipeline, download script, tests, manifest, and per-image CSV.
No deep learning, no model weights. Suggested future work: calibrated physical
scale, artifact rejection, a stronger classical segmentation baseline and a
proper patient-level external dataset study with clinicians. Do not claim any
melanoma accuracy until diagnosis labels and a valid study support it.
