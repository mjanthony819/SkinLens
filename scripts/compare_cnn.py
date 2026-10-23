"""Fair same-split comparison: frozen ImageNet backbone + linear probe vs the SkinLens hand-set score.

Uses the SAME 500-image HAM10000 subset as scripts/validate_score.py (seed 2026: 150 melanoma +
350 other), the SAME AUC implementation and bootstrap, and a LESION-GROUPED ~70/30 split built
from the official ISIC2018_Task3_Training_LesionGroupings.csv so that no lesion_id appears in
both train and test.

Honest scope:
- The split is grouped by lesion, NOT by patient. The official grouping file carries no patient
  identifier, so different lesions from the same patient may still appear on both sides.
- One small enriched sample (500 images, 30% melanoma vs ~11% in the full dataset). AUC does not
  depend on prevalence, but this is not clinical accuracy and not an external validation.
- The CNN here is deliberately conservative: a frozen torchvision mobilenet_v3_small
  (IMAGENET1K_V1) backbone plus a logistic-regression probe. It is a lower bound on what a CNN
  can do, not an optimised model.
"""
import argparse, csv, io, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from remotezip import RemoteZip
from skinlens.pipeline import analyze, VERSION

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://isic-archive.s3.amazonaws.com/challenges/2018/'
IMG_URL = BASE + 'ISIC2018_Task3_Training_Input.zip'
GT_URL = BASE + 'ISIC2018_Task3_Training_GroundTruth.zip'
GT_NAME = 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
GROUPS_URL = BASE + 'ISIC2018_Task3_Training_LesionGroupings.csv'


def auc(score, y):
    """Rank-based (Mann-Whitney) AUC with average ranks for ties. Identical to validate_score.py."""
    score = np.asarray(score, float); y = np.asarray(y, bool)
    order = np.argsort(score, kind='mergesort'); s = score[order]; ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j+1] == s[i]: j += 1
        ranks[i:j+1] = (i+j)/2 + 1; i = j+1
    r = np.empty(len(s)); r[order] = ranks
    n1 = y.sum(); n0 = (~y).sum()
    return float((r[y].sum() - n1*(n1+1)/2) / (n1*n0))


def boot_ci(score, y, n=2000, seed=2026):
    """Stratified bootstrap CI. Identical to validate_score.py."""
    rng = np.random.default_rng(seed); pos = np.flatnonzero(y); neg = np.flatnonzero(~y); out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(auc(score[idx], y[idx]))
    return np.quantile(out, [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--melanoma', type=int, default=150); p.add_argument('--other', type=int, default=350)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--data', type=Path, default=ROOT/'data'/'task3')
    p.add_argument('--output', type=Path, default=ROOT/'reports'/'validation')
    a = p.parse_args()
    a.data.mkdir(parents=True, exist_ok=True); a.output.mkdir(parents=True, exist_ok=True)

    # --- 1) same subset as validate_score.py (seed 2026) + lesion groupings ---
    import urllib.request
    groups_path = a.data / 'ISIC2018_Task3_Training_LesionGroupings.csv'
    if not groups_path.exists():
        urllib.request.urlretrieve(GROUPS_URL, groups_path)
    groups = {r['image']: r['lesion_id'] for r in csv.DictReader(open(groups_path))}
    with RemoteZip(GT_URL, timeout=90) as z:
        rows = list(csv.DictReader(io.StringIO(z.read(GT_NAME).decode())))
    mel = sorted(r['image'] for r in rows if float(r['MEL']) == 1)
    other = sorted(r['image'] for r in rows if float(r['MEL']) != 1)
    rng = np.random.default_rng(a.seed)
    ids = [(i, 1) for i in rng.choice(mel, a.melanoma, replace=False)] + \
          [(i, 0) for i in rng.choice(other, a.other, replace=False)]
    names_only = [i for i, _ in ids]
    y = np.array([l for _, l in ids], bool)
    with RemoteZip(IMG_URL, timeout=90) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.endswith('.jpg')}
        for k, (ident, _) in enumerate(ids):
            dest = a.data / f'{ident}.jpg'
            if not dest.exists():
                for t in range(3):
                    try: dest.write_bytes(z.read(names[ident])); break
                    except Exception:
                        if t == 2: raise
                        time.sleep(2)
            if (k+1) % 100 == 0: print(f'download {k+1}/{len(ids)}', flush=True)

    # --- 2) classical SkinLens scores (cached) ---
    cs_path = a.data / 'classical_scores_cnn_cmp.npy'
    if cs_path.exists():
        classical = np.load(cs_path)
    else:
        out = []
        for k, ident in enumerate(names_only):
            f = analyze(np.array(Image.open(a.data/f'{ident}.jpg').convert('RGB')))['features']
            out.append(f.score)
            if (k+1) % 100 == 0: print(f'classical {k+1}/{len(ids)}', flush=True)
        classical = np.array(out); np.save(cs_path, classical)

    # --- 3) frozen ImageNet backbone features (cached) ---
    feat_path = a.data / 'cnn_features.npy'
    if feat_path.exists():
        feats = np.load(feat_path)
    else:
        import torch, torchvision as tv
        torch.set_num_threads(2)
        model = tv.models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()
        body = torch.nn.Sequential(model.features, model.avgpool)
        tf = tv.transforms.Compose([
            tv.transforms.Resize(256), tv.transforms.CenterCrop(224), tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        outs = []
        with torch.no_grad():
            for k, ident in enumerate(names_only):
                x = tf(Image.open(a.data/f'{ident}.jpg').convert('RGB')).unsqueeze(0)
                outs.append(body(x).flatten().numpy())
                if (k+1) % 100 == 0: print(f'features {k+1}/{len(ids)}', flush=True)
        feats = np.stack(outs); np.save(feat_path, feats)

    # --- 4) lesion-grouped ~70/30 split: no lesion_id in both train and test ---
    g = np.array([groups.get(i, i) for i in names_only])
    uniq_groups = np.unique(g)
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        order = rng.permutation(uniq_groups)
        test_groups = set(order[:int(round(0.30 * len(uniq_groups)))])
        t = np.array([gg in test_groups for gg in g])
        score = abs(y[t].mean() - y.mean()) + abs(t.mean() - 0.30)
        if best is None or score < best[0]:
            best = (score, seed, t)
    _, split_seed, test_mask = best
    test = np.flatnonzero(test_mask); train = np.flatnonzero(~test_mask)

    # --- 5) logistic-regression probe on frozen features ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(feats[train])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(feats[train]), y[train])
    cnn_score = clf.decision_function(sc.transform(feats))

    # --- 6) verification: split disjointness, grouping-file facts, paired bootstrap on the difference ---
    overlap = set(g[train]) & set(g[test])
    from collections import Counter
    full_counts = Counter(groups.values())
    multi_full = sum(1 for v in full_counts.values() if v > 1)
    multi_sub = sum(1 for v in Counter(g).values() if v > 1)
    header = open(groups_path).readline().strip()
    yt, ct, nt = y[test], classical[test], cnn_score[test]
    obs_diff = auc(nt, yt) - auc(ct, yt)
    rng = np.random.default_rng(2026)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(yt), len(yt))
        if yt[idx].sum() in (0, len(yt)): continue
        diffs.append(auc(nt[idx], yt[idx]) - auc(ct[idx], yt[idx]))
    diffs = np.array(diffs)
    diff_ci = np.quantile(diffs, [.025, .975]).tolist()
    p_le0 = float((diffs <= 0).mean())

    res = {
      'pipeline_version': VERSION,
      'purpose': 'Fair same-split comparison of the SkinLens hand-set score vs a conservative CNN baseline (frozen ImageNet backbone + logistic-regression probe).',
      'dataset': 'ISIC 2018 Task 3 training (HAM10000) ground truth, melanoma (MEL) vs all other classes',
      'subset': f'seed {a.seed}; {a.melanoma} melanoma + {a.other} non-melanoma, same subset as reports/validation/score_validation.json',
      'n': len(y), 'n_train': int(len(train)), 'n_test': int(len(test)),
      'test_melanoma': int(yt.sum()),
      'split': f'LESION-GROUPED ~70/30 via ISIC2018_Task3_Training_LesionGroupings.csv (best of 200 seeded group shuffles, winning seed {split_seed}); verified: no lesion_id appears in both train and test',
      'classical_score_on_test': {'auc': auc(ct, yt), 'auc_bootstrap_95ci': boot_ci(ct, yt)},
      'cnn_linear_probe_on_test': {'auc': auc(nt, yt), 'auc_bootstrap_95ci': boot_ci(nt, yt)},
      'sanity_classical_on_all500': auc(classical, y),
      'paired_bootstrap_on_difference': {
        'observed_auc_difference': obs_diff,
        'diff_95ci': diff_ci,
        'p_diff_le_0': p_le0,
        'method': '2000 bootstrap resamples of the 150 test images, same resample applied to both scorers; draws with a single class skipped',
      },
      'split_verification': {
        'train_test_lesion_group_overlap': sorted(overlap),
        'split_disjoint_by_lesion': len(overlap) == 0,
        'official_grouping_file_header': header,
        'patient_id_available': 'patient' in header.lower(),
        'lesions_with_multiple_images_full_dataset': multi_full,
        'lesions_with_multiple_images_in_subset': multi_sub,
      },
      'cnn_baseline': 'torchvision mobilenet_v3_small, weights IMAGENET1K_V1 (ImageNet-1k pretrained, frozen), 576-d avgpool features + StandardScaler + LogisticRegression(C=1, max_iter=2000). Resize 256, center crop 224, ImageNet normalisation.',
      'scope': [
        'The split is grouped by lesion_id, NOT by patient: the official grouping file carries no patient identifier, so different lesions from the same patient may appear on both sides.',
        'One small enriched sample (500 images, 30% melanoma). This is not clinical accuracy and not an external validation.',
        'The frozen-backbone linear probe is a deliberately conservative CNN baseline; this result says nothing about what an optimised or fine-tuned model would achieve.',
      ],
      'bootstrap': 'individual CIs: 2000 stratified resamples, seed 2026 (identical code to scripts/validate_score.py)',
    }
    out_path = a.output / 'cnn_comparison.json'
    out_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
