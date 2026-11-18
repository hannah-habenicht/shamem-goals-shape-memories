"""Helper functions for encoding_bias.py: logging, loading the BIDS
derivatives, running R, and plotting. The analysis steps themselves (gates,
bias scores, models) are in encoding_bias.py."""
import contextlib
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Logging ────────────────────────────────────────────────────────

class Tee:
    """
    Duplicate stdout to a UTF-8 log file, plus one log file per ET measure
    (<out_dir>/log_encoding_ET_bias_<measure>.txt).

    Output written outside measure_scope() goes to the main log and every
    per-measure log, so each per-measure file reads on its own. Output
    inside `with log.measure_scope(meas):` goes only to that measure's file.
    """
    def __init__(self, path, measure_names, out_dir):
        self._file = open(path, 'w', encoding='utf-8')
        self._stdout = sys.__stdout__
        self._measure_files = {
            m: open(os.path.join(out_dir, f'log_encoding_ET_bias_{m}.txt'),
                    'w', encoding='utf-8')
            for m in measure_names
        }
        self._current_measure = None

    def write(self, data):
        self._stdout.write(data)
        if self._current_measure is not None:
            self._measure_files[self._current_measure].write(data)
        else:
            self._file.write(data)
            for f in self._measure_files.values():
                f.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()
        for f in self._measure_files.values():
            f.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()
        for f in self._measure_files.values():
            f.close()

    @contextlib.contextmanager
    def measure_scope(self, measure):
        prev = self._current_measure
        self._current_measure = measure
        try:
            yield
        finally:
            self._current_measure = prev


# ── Loading the BIDS derivatives ───────────────────────────────────

MOVIE_RE = re.compile(r'movie_(\d+)')

# Each participant watched 3 blocks x 6 films = 18 films (one run per film).
N_RUNS_PER_SUBJECT = 18

# Runs a loader could not use. The loaders record (file, reason) here
# instead of dropping a run silently; load_et_data() prints the list.
_SKIPPED_RUNS = []


def skip(fpath: str, reason: str, log: list = _SKIPPED_RUNS):
    """Record why a run was not used and return None (the loaders' 'no data')."""
    log.append((os.path.basename(fpath), reason))
    return None


def report_skipped(log: list, what: str):
    """Print a WARNING block listing every run in `log`, if any."""
    if not log:
        print(f"  All {what} files were used.")
        return
    print(f"  WARNING: {len(log)} {what} file(s) could not be used:")
    for fname, reason in sorted(log):
        print(f"    {fname}: {reason}")


def _report_incomplete(df: pd.DataFrame, value_col: str, what: str):
    """Print a WARNING for each participant with fewer than 18 usable runs."""
    # Count per participant seen in any file, so that someone with no usable
    # runs of this kind is listed with 0 instead of being left out.
    n = (df.dropna(subset=[value_col])
           .drop_duplicates(subset=['sub_id', 'block', 'film_id'])
           .groupby('sub_id').size()
           .reindex(sorted(df['sub_id'].unique()), fill_value=0))
    short = n[n < N_RUNS_PER_SUBJECT]
    if short.empty:
        print(f"  {what}: all {len(n)} participants have "
              f"{N_RUNS_PER_SUBJECT} runs.")
        return
    print(f"  WARNING: {what}: {len(short)} participant(s) have fewer than "
          f"{N_RUNS_PER_SUBJECT} usable runs (some may be excluded later):")
    for sub, k in short.items():
        print(f"    sub-{int(sub):03d}: {k} runs")


def read_sidecar_columns(tsv_path: str):
    """Column names for a header-less BIDS tsv.gz, from its JSON sidecar."""
    json_path = tsv_path[:-7] + '.json'
    with open(json_path, 'r', encoding='utf-8') as fh:
        return json.load(fh)['Columns']


def parse_bids_ids(fname: str):
    """Extract (sub_id, session) as ints from a BIDS filename, or None."""
    m = re.match(r'^sub-(\d+)_ses-(\d+)_', fname)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _aggregate_fixation(fpath: str):
    """
    Load one fixations physioevents TSV.gz (one run = one film viewing),
    filter valid fixations, return aggregated counts/durations.
    """
    ids = parse_bids_ids(os.path.basename(fpath))
    if ids is None:
        return skip(fpath, 'filename has no sub-/ses- ID')
    sub_id, session = ids

    try:
        cols = read_sidecar_columns(fpath)
        df = pd.read_csv(fpath, sep='\t', header=None, names=cols,
                         compression='gzip', low_memory=False)
    except Exception as e:
        return skip(fpath, f'unreadable ({type(e).__name__}: {e})')

    if df.empty:
        return skip(fpath, 'file is empty')
    if 'Category_Start' not in df.columns:
        return skip(fpath, 'no Category_Start column')

    # Film and block come from the data columns, not the filename
    m_movie = MOVIE_RE.search(str(df['Movie'].iloc[0]))
    block = pd.to_numeric(df['block'], errors='coerce').dropna()
    if m_movie is None or block.empty:
        return skip(fpath, 'no film or block label')
    film_id = int(m_movie.group(1))
    block = int(block.iloc[0])

    # OutOfBounds parses as bool from the TSV; go via str to be robust.
    # Counted here (before filtering) so a per-participant OOB rate can be
    # tracked as a tracking-quality signal (see Section 1B).
    oob = df['OutOfBounds'].astype(str).str.strip().str.lower()
    n_rows_total = len(df)
    n_rows_oob   = int((oob == 'true').sum())
    df = df[oob == 'false']
    if df.empty:
        return skip(fpath, 'all fixations out of bounds')

    # physioevents duration is in seconds; convert to ms
    duration_ms = pd.to_numeric(df['duration'], errors='coerce').fillna(0) * 1000.0

    cat = df['Category_Start'].astype(str).str.lower().str.strip()
    face_mask   = cat == 'personal'
    place_mask  = cat == 'spatial'
    others_mask = cat == 'others'
    object_mask = cat == 'object'
    # Valid (in-bounds) rows whose category doesn't match any of the four
    # known labels. ~0% for fixations, but kept for symmetry with samples,
    # where it is material (see _aggregate_sample).
    unassigned_mask = ~(face_mask | place_mask | others_mask | object_mask)

    return {
        'sub_id': sub_id, 'session': session, 'block': block, 'film_id': film_id,
        'face_fix_count':   int(face_mask.sum()),
        'place_fix_count':  int(place_mask.sum()),
        'others_fix_count': int(others_mask.sum()),
        'object_fix_count': int(object_mask.sum()),
        'unassigned_fix_count': int(unassigned_mask.sum()),
        'face_fix_dur_ms':   float(duration_ms[face_mask].sum()),
        'place_fix_dur_ms':  float(duration_ms[place_mask].sum()),
        'others_fix_dur_ms': float(duration_ms[others_mask].sum()),
        'object_fix_dur_ms': float(duration_ms[object_mask].sum()),
        'unassigned_fix_dur_ms': float(duration_ms[unassigned_mask].sum()),
        # All-category valid totals - used by Section 11's category
        # breakdown (viewing time share per category), not by the
        # face/place proportions above (those use face+place only).
        'valid_fix_count':  int(len(df)),
        'valid_fix_dur_ms': float(duration_ms.sum()),
        'fix_rows_total': n_rows_total,
        'fix_rows_oob':   n_rows_oob,
    }


def _aggregate_sample(fpath: str):
    """
    Load one eye1 physio TSV.gz (raw gaze samples for one film viewing),
    filter valid samples, return per-category sample counts.

    Only the eye1 file is read: the pipeline assigns one pixel per sample
    (left eye preferred, right as fallback) and writes identical category
    columns to the eye1 and eye2 files, so reading both would double count.
    """
    ids = parse_bids_ids(os.path.basename(fpath))
    if ids is None:
        return skip(fpath, 'filename has no sub-/ses- ID')
    sub_id, session = ids

    try:
        cols = read_sidecar_columns(fpath)
        df = pd.read_csv(fpath, sep='\t', header=None, names=cols,
                         compression='gzip', low_memory=False)
    except Exception as e:
        return skip(fpath, f'unreadable ({type(e).__name__}: {e})')

    if df.empty:
        return skip(fpath, 'file is empty')
    if 'category' not in df.columns:
        return skip(fpath, 'no category column')

    # Film comes from the chunk key (samples carry no Movie column)
    m_movie = MOVIE_RE.search(str(df['source_chunk_key'].iloc[0]))
    block = pd.to_numeric(df['block'], errors='coerce').dropna()
    if m_movie is None or block.empty:
        return skip(fpath, 'no film or block label')
    film_id = int(m_movie.group(1))
    block = int(block.iloc[0])

    oob = df['out_of_bounds'].astype(str).str.strip().str.lower()
    n_rows_total = len(df)
    n_rows_oob   = int((oob == 'true').sum())
    df = df[oob == 'false']
    if df.empty:
        return skip(fpath, 'all samples out of bounds')

    cat = df['category'].astype(str).str.lower().str.strip()
    face_mask   = cat == 'personal'
    place_mask  = cat == 'spatial'
    others_mask = cat == 'others'
    object_mask = cat == 'object'
    # Valid (not out-of-bounds) samples with no category: samples without a
    # gaze position (blinks, brief tracking loss), which therefore have no
    # pixel to look up. About 2-4% of samples per group (checked against the
    # raw files, 2026-11-06). Tracked explicitly so Section 11's category
    # shares sum to 100% instead of silently omitting this slice.
    unassigned_mask = ~(face_mask | place_mask | others_mask | object_mask)

    return {
        'sub_id': sub_id, 'session': session, 'block': block, 'film_id': film_id,
        'face_sample_count':       int(face_mask.sum()),
        'place_sample_count':      int(place_mask.sum()),
        'others_sample_count':     int(others_mask.sum()),
        'object_sample_count':     int(object_mask.sum()),
        'unassigned_sample_count': int(unassigned_mask.sum()),
        # All-category valid total - used by Section 11's category breakdown.
        'valid_sample_count': int(len(df)),
        'samp_rows_total': n_rows_total,
        'samp_rows_oob':   n_rows_oob,
    }


def safe_prop(num: pd.Series, denom: pd.Series) -> pd.Series:
    """num / denom, returning NaN when denominator is 0. Used both for
    face/(face+place) proportions and, in Section 11, for a single
    category's share of the all-category valid total."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, num / denom, np.nan)


def load_et_data(path_et: str, n_workers: int = 8) -> pd.DataFrame:
    """
    Scan BIDS derivatives (ses-01), aggregate all fixation physioevents and
    eye1 sample physio files, and return a wide DataFrame with one row per
    movie-clip viewing.
    """
    fix_files = sorted(glob.glob(os.path.join(
        path_et, 'sub-*', 'ses-01', 'eyetrack',
        '*_recording-fixations_physioevents.tsv.gz')))
    samp_files = sorted(glob.glob(os.path.join(
        path_et, 'sub-*', 'ses-01', 'eyetrack',
        '*_recording-eye1_physio.tsv.gz')))

    print(f"  Found {len(fix_files)} fixation files, {len(samp_files)} sample files.")

    fix_rows, samp_rows = [], []

    print("  Loading fixation files…")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = ex.map(_aggregate_fixation, fix_files)
        for result in tqdm(futures, total=len(fix_files), ncols=80):
            if result is not None:
                fix_rows.append(result)

    print("  Loading sample files…")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = ex.map(_aggregate_sample, samp_files)
        for result in tqdm(futures, total=len(samp_files), ncols=80):
            if result is not None:
                samp_rows.append(result)

    print(f"  Parsed: {len(fix_rows)} fixation records, {len(samp_rows)} sample records.")
    report_skipped(_SKIPPED_RUNS, 'fixation/sample')

    df_fix  = pd.DataFrame(fix_rows)
    df_samp = pd.DataFrame(samp_rows)

    key_cols = ['sub_id', 'session', 'block', 'film_id']
    df_et = df_fix.merge(df_samp, on=key_cols, how='outer')

    # Missing files (no run on disk) show up here too, not only skipped ones
    _report_incomplete(df_et, 'face_fix_count',    'Fixations')
    _report_incomplete(df_et, 'face_sample_count', 'Samples')

    # Proportions: face / (face + place); objects excluded from denominator
    df_et['face_count_prop']    = safe_prop(
        df_et['face_fix_count'],
        df_et['face_fix_count'] + df_et['place_fix_count'])
    df_et['face_duration_prop'] = safe_prop(
        df_et['face_fix_dur_ms'],
        df_et['face_fix_dur_ms'] + df_et['place_fix_dur_ms'])
    df_et['face_sample_prop']   = safe_prop(
        df_et['face_sample_count'],
        df_et['face_sample_count'] + df_et['place_sample_count'])

    # Category shares (Section 11 only): each category's count/duration/
    # sample count as a fraction of the ALL-category valid total (personal +
    # spatial + others + object + unassigned), not just face+place. These
    # five shares sum to 1 for a given measure (modulo NaN rows with 0 valid
    # data). 'unassigned' is ~0 for fixations but material for samples (see
    # _aggregate_sample) - kept in both for a consistent, complete breakdown.
    for cat_name, count_col, dur_col in [
        ('face',       'face_fix_count',       'face_fix_dur_ms'),
        ('place',      'place_fix_count',      'place_fix_dur_ms'),
        ('others',     'others_fix_count',     'others_fix_dur_ms'),
        ('object',     'object_fix_count',     'object_fix_dur_ms'),
        ('unassigned', 'unassigned_fix_count', 'unassigned_fix_dur_ms'),
    ]:
        df_et[f'{cat_name}_count_share'] = safe_prop(df_et[count_col], df_et['valid_fix_count'])
        df_et[f'{cat_name}_dur_share']   = safe_prop(df_et[dur_col],   df_et['valid_fix_dur_ms'])

    for cat_name, samp_col in [
        ('face',       'face_sample_count'),
        ('place',      'place_sample_count'),
        ('others',     'others_sample_count'),
        ('object',     'object_sample_count'),
        ('unassigned', 'unassigned_sample_count'),
    ]:
        df_et[f'{cat_name}_sample_share'] = safe_prop(df_et[samp_col], df_et['valid_sample_count'])

    return df_et


def load_behav_data(path_beh: str, path_cond: str) -> pd.DataFrame:
    """
    Load behavioural CSV and return a condition lookup table:
    one row per (Participant_ID, session, block, film_id) with job/timing labels.

    job and timing are taken from the corrected conditions.csv (not from the
    behavioural CSV, which contained student data entry errors). The two files
    are joined on participant id; the old job/timing columns are discarded.

    conditions.csv columns used:
      id   - participant ID
      type - instruction timing (1=Pro, 2=Retro, 3=Neutral), renamed to 'timing'
      job  - role assignment  (1=Spy, 2=Criminal, 3=Neutral)
    """
    df = pd.read_csv(path_beh)
    df = df.rename(columns={'id': 'Participant_ID', 'filmID': 'film_id'})
    df = df.drop(columns=['job', 'timing'])   # discard potentially wrong values

    cond = pd.read_csv(path_cond)
    cond = cond.rename(columns={'id': 'Participant_ID', 'type': 'timing'})
    cond['job']    = cond['job'].map({1: 'Spy', 2: 'Criminal', 3: 'Neutral'})
    cond['timing'] = cond['timing'].map({1: 'Pro', 2: 'Retro', 3: 'Neutral'})
    cond = cond[['Participant_ID', 'job', 'timing']]

    df = df.merge(cond, on='Participant_ID', how='left')

    lookup_cols = ['Participant_ID', 'session', 'block', 'film_id', 'job', 'timing']
    lookup = (df[lookup_cols]
              .drop_duplicates(subset=['Participant_ID', 'session', 'block', 'film_id'])
              .reset_index(drop=True))
    return lookup


# ── Running R ──────────────────────────────────────────────────────

def run_r(template: str, data: pd.DataFrame, rscript: str, **fmt):
    """Write `data` to a temporary CSV, fill `template` with csv_path=<that
    file> and **fmt, run it with Rscript and return the CompletedProcess.
    Temporary files are always removed."""
    csv_path = r_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            csv_path = f.name
            data.to_csv(f, index=False)
        with tempfile.NamedTemporaryFile(suffix='.R', delete=False, mode='w', encoding='utf-8') as rf:
            r_path = rf.name
            rf.write(template.format(csv_path=csv_path.replace('\\', '/'), **fmt))
        return subprocess.run([rscript, '--vanilla', r_path], capture_output=True, text=True)
    finally:
        for p in (csv_path, r_path):
            if p and os.path.exists(p):
                os.unlink(p)


def print_r_output(result):
    """Print R's stdout, and any stderr lines that are not routine package
    messages or warnings."""
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        errors = [ln for ln in result.stderr.splitlines()
                  if ln.strip()
                  and not ln.startswith('Loading')
                  and not re.match(r'^\d+: package .* was built under R', ln)
                  and 'Warning' not in ln
                  and 'kenward-roger' not in ln
                  and 'boundary (singular) fit' not in ln
                  and 'NOTE:' not in ln]
        if errors:
            tag = '[R error]' if result.returncode != 0 else '[R warning]'
            print(f"  {tag}", '\n'.join(errors))


# ── Plotting ───────────────────────────────────────────────────────

def raincloud(ax, data, pos, color, seed=42, point_size=10):
    """
    Raincloud for one group at x=pos (ggplot style):
      - half-violin (untrimmed KDE, bandwidth x1.2) to the LEFT of pos,
        with black lines at the 25th and 75th percentiles
      - jittered raw points to the RIGHT of pos (pos + 0.25 +/- 0.1)
    The centre point is drawn by the caller.
    """
    from scipy.stats import gaussian_kde
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return
    if len(data) >= 2 and np.ptp(data) > 0:
        kde = gaussian_kde(data, bw_method='silverman')
        kde.set_bandwidth(kde.factor * 1.2)            # ggplot adjust = 1.2
        bw  = kde.factor * data.std(ddof=1)
        ys  = np.linspace(data.min() - 3 * bw, data.max() + 3 * bw, 300)   # trim = FALSE
        dens = kde(ys)
        half_w = 0.4 * dens / dens.max()
        ax.fill_betweenx(ys, pos - half_w, pos, facecolor=color,
                         edgecolor='none', alpha=0.8, zorder=1)
        for q in np.percentile(data, [25, 75]):
            w = 0.4 * kde(q)[0] / dens.max()
            ax.plot([pos - w, pos], [q, q], color='black', linewidth=0.7, zorder=2)
    jitter = np.random.default_rng(seed).uniform(-0.1, 0.1, size=len(data))
    ax.scatter(pos + 0.25 + jitter, data, color=color, alpha=0.7, s=point_size,
               linewidths=0, zorder=3)


def centre_point(ax, x, y):
    """Black centre point (group mean)."""
    ax.scatter([x], [y], color='black', marker='o', s=40, zorder=6)


def style_ax(ax):
    """theme_minimal look: no grid, no spines, no tick marks; solid 0 line."""
    ax.axhline(0, linestyle='-', color='black', linewidth=0.8, zorder=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0, colors='black')


def _p_label(p, show_trends):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    if p < .10 and show_trends: return '+'
    return None


def contrast_pairs(contr, pos_of, show_trends):
    """[(x1, x2, label)] for the significant rows of a contrast table (columns
    'contrast' as "A - B" and 'p.value'). pos_of maps a level name (e.g.
    'Spy_Pro' or '3') to its x position; pairs involving a level not on the
    panel are skipped."""
    pairs = []
    for _, row in contr.iterrows():
        lab = _p_label(row['p.value'], show_trends)
        a, b = [s.strip() for s in row['contrast'].split(' - ')]
        if lab and a in pos_of and b in pos_of:
            pairs.append((min(pos_of[a], pos_of[b]), max(pos_of[a], pos_of[b]), lab))
    return pairs


def sig_pairs(contr, family, by, pos_of, show_trends):
    """contrast_pairs() for one panel of a table with 'family' and 'by' columns."""
    sub = contr[(contr['family'] == family) & (contr['by'] == str(by))]
    return contrast_pairs(sub, pos_of, show_trends)


def draw_brackets(axes_and_pairs, fontsize):
    """Draw stacked significance brackets (plain line, stars
    above) on a row of sharey axes. Short brackets sit lowest; a bracket moves
    up a level if it would touch or overlap one already on that level."""
    y_lo, y_hi = axes_and_pairs[0][0].get_ylim()
    step = 0.09 * (y_hi - y_lo)
    n_levels = 0
    for ax, pairs in axes_and_pairs:
        levels = []   # per level: list of occupied (x1, x2)
        for x1, x2, lab in sorted(pairs, key=lambda p: p[1] - p[0]):
            lvl = next((i for i, occ in enumerate(levels)
                        if all(x2 < o1 or x1 > o2 for o1, o2 in occ)), len(levels))
            if lvl == len(levels):
                levels.append([])
            levels[lvl].append((x1, x2))
            y = y_hi + (lvl + 0.3) * step
            ax.plot([x1, x2], [y, y], color='black', linewidth=0.8, clip_on=False)
            ax.text((x1 + x2) / 2, y + 0.05 * step, lab, ha='center', va='bottom',
                    fontweight='bold', fontsize=fontsize)
        n_levels = max(n_levels, len(levels))
    if n_levels:
        axes_and_pairs[0][0].set_ylim(y_lo, y_hi + (n_levels + 0.3) * step)


def polar_interp(r_closed, n_cat, petal_width, petal_tip, offset=0.0, n_points=720):
    """Dense (theta, r) path for the radar plot: one pointed petal per
    category value. r(theta) = value * (1 - u**petal_tip), u = d / half_width,
    d = angular distance from the category axis; petal_width = fraction of
    each category's sector the petal spans."""
    vals = np.asarray(r_closed, dtype=float)[:n_cat]
    seg = 2 * np.pi / n_cat
    half_w = petal_width * seg / 2
    theta = np.linspace(0, 2 * np.pi, n_points + 1)
    k = np.round(theta / seg).astype(int) % n_cat          # nearest category axis
    d = np.abs((theta - k * seg + np.pi) % (2 * np.pi) - np.pi)
    shape = np.where(d < half_w, 1 - (d / half_w) ** petal_tip, 0.0)
    return theta + offset, vals[k] * shape
