"""One-time conversion of segmentation mask CSVs to grayscale label PNGs.

The mask CSVs in SEGMENTATION_DIR (e.g. <MASKS_DIR>/Biohackers/*_frame_0000.csv)
are ~4.6 MB text files that take ~130-280 ms each to parse. This script re-saves
each one as a lossless grayscale PNG (~24 KB, ~8 ms to decode) in a parallel
directory tree:

    <SEGMENTATION_DIR>_labels/<Movie>/<Movie>_frame_XXXX.png

The CSVs are read with header=0 (their first line is a column-index header),
so the PNGs contain the correctly aligned 1080x1920 label grid.

The script is resumable: existing PNGs are skipped, so it can be interrupted
and re-run. Run it once with the pipeline environment:

    python convert_masks_to_labels.py
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from pipeline_utils import SEGMENTATION_DIR

LABELS_DIR = SEGMENTATION_DIR.rstrip('\\/') + '_labels'


def convert_one(csv_path, png_path):
    """Convert a single mask CSV to a grayscale label PNG.

    Returns (csv_path, error_message_or_None).
    """
    try:
        labels = pd.read_csv(csv_path, delimiter=';', header=0, index_col=0).to_numpy()
        if labels.min() < 0 or labels.max() > 255:
            return csv_path, f'labels out of uint8 range ({labels.min()}..{labels.max()})'
        labels = labels.astype(np.uint8)
        if not cv2.imwrite(png_path, labels):
            return csv_path, 'cv2.imwrite failed'
        # verify lossless roundtrip before trusting the PNG
        back = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if not np.array_equal(back, labels):
            os.remove(png_path)
            return csv_path, 'roundtrip verification failed'
        return csv_path, None
    except Exception as e:
        return csv_path, str(e)


def collect_jobs():
    """Find all mask CSVs that do not have a label PNG yet."""
    jobs = []
    skipped = 0
    for movie_folder in sorted(os.listdir(SEGMENTATION_DIR)):
        src_dir = os.path.join(SEGMENTATION_DIR, movie_folder)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(LABELS_DIR, movie_folder)
        os.makedirs(dst_dir, exist_ok=True)
        for file_name in os.listdir(src_dir):
            if not file_name.endswith('.csv'):
                continue
            png_name = file_name[:-4] + '.png'
            png_path = os.path.join(dst_dir, png_name)
            if os.path.exists(png_path):
                skipped += 1
                continue
            jobs.append((os.path.join(src_dir, file_name), png_path))
    return jobs, skipped


if __name__ == '__main__':
    if not os.path.isdir(SEGMENTATION_DIR):
        raise SystemExit(f'Segmentation directory not found: {SEGMENTATION_DIR}')

    jobs, skipped = collect_jobs()
    print(f'Output directory: {LABELS_DIR}')
    print(f'{len(jobs)} CSVs to convert ({skipped} already converted, skipping)')
    if not jobs:
        print('Nothing to do.')
        raise SystemExit(0)

    errors = []
    done = 0
    start = time.time()
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(convert_one, csv_path, png_path) for csv_path, png_path in jobs]
        for future in as_completed(futures):
            csv_path, error = future.result()
            done += 1
            if error:
                errors.append((csv_path, error))
                print(f'ERROR {csv_path}: {error}')
            if done % 250 == 0 or done == len(jobs):
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(jobs) - done) / rate if rate > 0 else 0
                print(f'{done}/{len(jobs)} converted '
                      f'({rate:.1f} frames/s, ~{remaining / 60:.0f} min remaining)')

    print(f'Finished in {(time.time() - start) / 60:.1f} min, {len(errors)} errors')
    if errors:
        print('Failed files:')
        for csv_path, error in errors:
            print(f'  {csv_path}: {error}')