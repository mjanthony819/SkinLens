# Data rights and attribution

Original project code is MIT. Dataset content is not MIT.

The bundled four downsampled samples in `samples/` and image-derived figures in
`reports/` contain ISIC 2018 Task 1 training imagery. The official aggregate
ISIC 2018 training release states CC-BY-NC (noncommercial). Keep attribution
and these restrictions when sharing the samples or figures. Review the current
provider terms before a public or commercial deployment; a hackathon entry does
not waive noncommercial restrictions. The project does not grant rights to the data.

Sources:
- https://challenge.isic-archive.com/data/ (2018 release and attribution)
- https://figshare.com/articles/dataset/ISIC2018_Task1-2_Training_Input_zip/22698379
- https://challenge.isic-archive.com/landing/2018/45/ (mask provenance)

Required aggregate attribution:

HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology,
Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161

MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006;
https://arxiv.org/abs/1902.03368

Full citations:

1. Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi,
Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris,
Michael Marchetti, Harald Kittler, Allan Halpern. "Skin Lesion Analysis Toward
Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging
Collaboration (ISIC)", 2018. https://arxiv.org/abs/1902.03368
2. Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin lesions.
Scientific Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

Samples were resized with preserved aspect ratio, JPEG re-encoded, and selected
from the development split. IDs are recorded in `samples/credits.json`.
Evaluation figures were resized and annotated with predicted contours.
No diagnosis is assigned to any displayed source image.
