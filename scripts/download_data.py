"""Download deterministic matched ISIC 2018 pairs, using HTTP Range ZIP access."""
import argparse, concurrent.futures, hashlib, json, random, time
from pathlib import Path
from remotezip import RemoteZip

BASE = 'https://isic-archive.s3.amazonaws.com/challenges/2018/'
IMAGE_URL = BASE + 'ISIC2018_Task1-2_Training_Input.zip'
MASK_URL = BASE + 'ISIC2018_Task1_Training_GroundTruth.zip'
ROOT = Path(__file__).resolve().parents[1]

def fetch_chunk(pairs, root):
    for attempt in range(3):
        try:
            with RemoteZip(IMAGE_URL, timeout=90) as iz, RemoteZip(MASK_URL, timeout=90) as mz:
                for ident, image_name, mask_name in pairs:
                    for z, name, dest in [(iz, image_name, root/'images'/f'{ident}.jpg'),
                                          (mz, mask_name, root/'masks'/f'{ident}.png')]:
                        if not dest.exists():
                            content = z.read(name)
                            tmp = dest.with_suffix('.part'); tmp.write_bytes(content); tmp.replace(dest)
            return
        except Exception:
            if attempt == 2: raise
            time.sleep(2**attempt)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--count',type=int,default=300); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--workers',type=int,default=6); p.add_argument('--output',type=Path,default=ROOT/'data'); a=p.parse_args()
    if not 51 <= a.count <= 2594: p.error('--count must be 51..2594 (50 development + remaining evaluation)')
    for sub in ['images','masks']: (a.output/sub).mkdir(parents=True,exist_ok=True)
    with RemoteZip(IMAGE_URL, timeout=90) as z: images={Path(n).stem:n for n in z.namelist() if n.lower().endswith('.jpg')}
    with RemoteZip(MASK_URL, timeout=90) as z: masks={Path(n).stem.removesuffix('_segmentation'):n for n in z.namelist() if n.endswith('.png')}
    ids=sorted(set(images)&set(masks)); random.Random(a.seed).shuffle(ids); ids=ids[:a.count]
    pairs=[(i,images[i],masks[i]) for i in ids]
    chunks=[pairs[i::a.workers] for i in range(a.workers)]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for _ in pool.map(lambda c:fetch_chunk(c,a.output),chunks): print('Download chunk complete',flush=True)
    records=[]
    for idx,ident in enumerate(ids):
        records.append({'id':ident,'split':'development' if idx<50 else 'evaluation',
          'image_sha256':hashlib.sha256((a.output/'images'/f'{ident}.jpg').read_bytes()).hexdigest(),
          'mask_sha256':hashlib.sha256((a.output/'masks'/f'{ident}.png').read_bytes()).hexdigest()})
    manifest={'dataset':'ISIC 2018 Task 1 training subset','seed':a.seed,'image_url':IMAGE_URL,'mask_url':MASK_URL,
              'license':'CC-BY-NC 4.0; see DATA_LICENSE.md','records':records}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Ready: {len(records)} pairs; 50 development, {len(records)-50} evaluation')
if __name__=='__main__': main()
