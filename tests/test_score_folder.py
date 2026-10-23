"""Folder-scoring CLI: synthetic images, CSV round-trip, error rows instead of crashes."""
import csv, importlib.util, pathlib
import numpy as np
import cv2
from PIL import Image
from skinlens.pipeline import analyze


def _load_script(name):
    path = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sf = _load_script('score_folder')


def _circle():
    im = np.full((256, 256, 3), (215, 175, 150), np.uint8)
    cv2.circle(im, (128, 128), 55, (85, 45, 30), -1)
    return im


def test_find_images_filters_and_sorts(tmp_path):
    Image.fromarray(_circle()).save(tmp_path / 'b.png')
    Image.fromarray(_circle()).save(tmp_path / 'a.jpg')
    (tmp_path / 'notes.txt').write_text('not an image')
    (tmp_path / 'nested').mkdir()
    assert [p.name for p in sf.find_images(tmp_path)] == ['a.jpg', 'b.png']


def test_scores_match_single_image_pipeline(tmp_path):
    Image.fromarray(_circle()).save(tmp_path / 'circle.png')  # PNG is lossless: batch score must match the raw array exactly
    row = sf.score_paths(sf.find_images(tmp_path))[0]
    f = analyze(_circle())['features']
    assert row['file'] == 'circle.png'
    assert row['score'] == round(f.score, 1)
    assert row['band'] == f.band
    assert row['warnings'] == ';'.join(f.warnings)


def test_warned_image_is_unreliable_in_batch(tmp_path):
    Image.fromarray(np.full((100, 100, 3), 150, np.uint8)).save(tmp_path / 'blank.jpg')
    row = sf.score_paths(sf.find_images(tmp_path))[0]
    assert row['band'] == 'Unreliable'
    assert row['warnings']


def test_broken_file_becomes_error_row_not_crash(tmp_path):
    (tmp_path / 'broken.jpg').write_bytes(b'this is not an image')
    row = sf.score_paths(sf.find_images(tmp_path))[0]
    assert row['file'] == 'broken.jpg'
    assert row['band'] == 'Error'
    assert row['warnings']


def test_write_csv_round_trip(tmp_path):
    rows = [{'file': 'x.jpg', 'score': 12.3, 'band': 'Low', 'colors': 'light brown', 'warnings': ''}]
    out = tmp_path / 'out.csv'
    sf.write_csv(rows, out)
    back = list(csv.DictReader(out.open()))
    assert back == [{'file': 'x.jpg', 'score': '12.3', 'band': 'Low', 'colors': 'light brown', 'warnings': ''}]
