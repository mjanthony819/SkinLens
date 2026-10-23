"""Score a folder of unlabeled dermoscopy photos and write a CSV summary.

Unlike scripts/evaluate.py, no expert masks are needed: this is the same
visual-irregularity score the app shows, applied to a whole folder at once
(for example a phone's camera roll). Not a diagnosis; flagged rows should
not be trusted.
"""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from skinlens.pipeline import analyze

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
FIELDS = ['file', 'score', 'band', 'colors', 'warnings']

def find_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)

def score_paths(paths, clahe=False):
    """Score each image. A file that fails to load becomes an Error row; it never stops the batch."""
    rows = []
    for path in paths:
        row = {'file': Path(path).name, 'score': '', 'band': 'Error', 'colors': '', 'warnings': ''}
        try:
            image = np.array(Image.open(path).convert('RGB'))
            f = analyze(image, clahe)['features']
            row.update(score=round(f.score, 1), band=f.band,
                       colors=';'.join(f.colors), warnings=';'.join(f.warnings))
        except Exception as e:
            row['warnings'] = f'{type(e).__name__}: {e}'
        rows.append(row)
    return rows

def write_csv(rows, out):
    with open(out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--output', type=Path, default=None, help='CSV path (default: <folder>/skinlens_scores.csv)')
    p.add_argument('--clahe', action='store_true', help='Contrast-enhance like the app checkbox (colors are still measured without it).')
    a = p.parse_args()
    paths = find_images(a.folder)
    if not paths:
        raise SystemExit(f'No images found in {a.folder}')
    rows = score_paths(paths, a.clahe)
    out = a.output or a.folder / 'skinlens_scores.csv'
    write_csv(rows, out)
    counts = {}
    for r in rows:
        counts[r['band']] = counts.get(r['band'], 0) + 1
    print(f"{len(rows)} images scored -> {out} | " + ', '.join(f'{k}: {v}' for k, v in sorted(counts.items())))

if __name__ == '__main__':
    main()
