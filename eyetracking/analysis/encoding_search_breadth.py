# ==================================================================
# ENCODING - GAZE SEARCH BREADTH BY GROUP
# Observation from browsing individual gaze plots: criminals
# (particularly Crim_Pro) seem to look around more / explore more of
# the background than spies. This tests that impression quantitatively
# using scanning-extent measures that are independent of the existing
# FIGURES
#   n_fix            - fixations per film viewing
#   mean_fix_dur_ms  - mean duration of a single fixation (ms) - dwelling
#                      longer per glance vs moving on quickly
#   scanpath_px      - total distance traveled between consecutive
#                      fixations (temporal order) - how much ground the
#                      eyes covered
#   dispersion_px    - mean distance of fixations from their own centroid
#                      (spatial coverage, independent of pacing). Modelled
#                      and logged because the paper reports it (null
#                      result), but not plotted.
#
# Only Control / Spy_Pro / Crim_Pro are compared (Group_Enc), matching
# encoding_bias.py's convention: Retro participants
# don't learn their role until after encoding, so they're not
# expected to differ from Neutral during encoding and are collapsed
# into Control.
#
# Groups, exclusions, and Category_Start face/place split mirror
# encoding_bias.py so results are directly comparable.
# EXCLUDE_SUBJECTS is copied from that script's own verified run
# (RESULTS_DIR/eyetracking_results/log_encoding_ET_bias.txt) rather than recomputed, since the
# QC gate there also uses raw 500Hz sample files this script doesn't
# load.
#
# GROUP SIGNIFICANCE TEST
# ------------------------
# Mirrors encoding_bias.py's Section 6 model: for each
# metric,
#   lmer(metric ~ Group_Enc * block_fac + film + (1|sub_id)
#        + (1|sub_id:block_fac) + (1|sub_id:film))
# fit via R (lme4/lmerTest/emmeans; REML, Satterthwaite df), with
# Tukey-adjusted pairwise post-hoc contrasts (Spy_Pro vs Control,
# Crim_Pro vs Control, Spy_Pro vs Crim_Pro) and standardized effect sizes.
# Group is between-subjects; block and film are within-subjects and fully
# crossed (each participant sees all 6 films in each of 3 blocks), so:
#   (1|sub_id:block_fac) - people differ in how they change across blocks
#                          (without it Group x block is tested against the
#                          trial-level residual and is anti-conservative)
#   (1|sub_id:film)      - the same person sees each film 3 times
#   film (fixed)         - film-specific scanning demands; 6 films only
#
# Helper functions (logging, file loading, running R, plotting) are in
# helper.py.
# ==================================================================

import os
import glob
import shutil
import sys
import numpy as np
import pandas as pd
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from helper import (Tee, MOVIE_RE, N_RUNS_PER_SUBJECT, read_sidecar_columns,
                    parse_bids_ids, skip, report_skipped, load_behav_data,
                    run_r, print_r_output, raincloud, contrast_pairs, draw_brackets)

PATH_ET   = config.ET_DERIVATIVES
PATH_BEH  = config.BEHAV_ENCODING_CSV
PATH_COND = config.CONDITIONS_CSV
# The paper figure goes to RESULTS_DIR; logs and contrast CSVs to RESULTS_DIR/eyetracking_results/.
PATH_PAPER = config.RESULTS_DIR
PATH_OUT   = os.path.join(PATH_PAPER, "eyetracking_results")
os.makedirs(PATH_OUT, exist_ok=True)

# Everything printed also goes to log_encoding_ET_search_breadth.txt
_log = Tee(os.path.join(PATH_OUT, 'log_encoding_ET_search_breadth.txt'), [], PATH_OUT)
sys.stdout = _log

# Path to Rscript - R needs lme4, lmerTest and emmeans installed
RSCRIPT = config.RSCRIPT
HAS_R = shutil.which(RSCRIPT) is not None

# Verified EXCLUDE_SUBJECTS from a real run of encoding_bias.py
# (RESULTS_DIR/eyetracking_results/log_encoding_ET_bias.txt, "Combined EXCLUDE_SUBJECTS" line).
EXCLUDE_SUBJECTS = [7, 14, 16, 25, 48, 52, 58, 59, 68, 74, 79, 84, 85, 90, 94, 95]

GROUP_PALETTE = {
    'Control':  '#A6A6A6',
    'Spy_Pro':  '#0072B2',
    'Crim_Pro': '#D55E00',
}
GROUP_ORDER = list(GROUP_PALETTE.keys())

# Runs the loader could not use, as (file, reason); printed after loading
# instead of dropping runs silently.
_SKIPPED_RUNS = []


def _load_fixation_run(fpath: str):
    """One run (= one film viewing): return trial-level scanning-breadth metrics."""
    ids = parse_bids_ids(os.path.basename(fpath))
    if ids is None:
        return skip(fpath, 'filename has no sub-/ses- ID', _SKIPPED_RUNS)
    sub_id, session = ids

    try:
        cols = read_sidecar_columns(fpath)
        df = pd.read_csv(fpath, sep='\t', header=None, names=cols,
                         compression='gzip', low_memory=False)
    except Exception as e:
        return skip(fpath, f'unreadable ({type(e).__name__}: {e})', _SKIPPED_RUNS)

    if df.empty:
        return skip(fpath, 'file is empty', _SKIPPED_RUNS)
    if 'Category_Start' not in df.columns:
        return skip(fpath, 'no Category_Start column', _SKIPPED_RUNS)

    m_movie = MOVIE_RE.search(str(df['Movie'].iloc[0]))
    block = pd.to_numeric(df['block'], errors='coerce').dropna()
    if m_movie is None or block.empty:
        return skip(fpath, 'no film or block label', _SKIPPED_RUNS)
    film_id = int(m_movie.group(1))
    block = int(block.iloc[0])

    oob = df['OutOfBounds'].astype(str).str.strip().str.lower()
    df = df[oob == 'false'].copy()
    if df.empty:
        return skip(fpath, 'all fixations out of bounds', _SKIPPED_RUNS)

    for c in ['tStart', 'tEnd', 'xAvg', 'yAvg']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['duration_ms'] = pd.to_numeric(df['duration'], errors='coerce') * 1000.0
    df = df.dropna(subset=['tStart', 'tEnd', 'xAvg', 'yAvg', 'duration_ms']).sort_values('tStart')
    if df.empty:
        return skip(fpath, 'no fixation with complete timing and position', _SKIPPED_RUNS)

    n = len(df)
    xs, ys = df['xAvg'].values, df['yAvg'].values
    if n >= 2:
        scanpath = float(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2).sum())
    else:
        scanpath = 0.0
    mean_fix_dur_ms = float(df['duration_ms'].mean())
    cx, cy = xs.mean(), ys.mean()
    dispersion = float(np.sqrt((xs - cx)**2 + (ys - cy)**2).mean())

    return {
        'sub_id': sub_id, 'session': session, 'block': block, 'film_id': film_id,
        'n_fix': n, 'mean_fix_dur_ms': mean_fix_dur_ms,
        'scanpath_px': scanpath, 'dispersion_px': dispersion,
    }


# ==================================================================
# 1. Load fixation runs
# ==================================================================
print("Loading encoding fixation files...")
fix_files = sorted(glob.glob(os.path.join(
    PATH_ET, 'sub-*', 'ses-01', 'eyetrack',
    '*_recording-fixations_physioevents.tsv.gz')))
print(f"  Found {len(fix_files)} files.")

rows = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for res in tqdm(ex.map(_load_fixation_run, fix_files), total=len(fix_files), ncols=80):
        if res is not None:
            rows.append(res)

report_skipped(_SKIPPED_RUNS, 'fixation')

df_trial = pd.DataFrame(rows)

# Participants with fewer than 18 usable runs (missing or skipped files)
# Counted per participant with any fixation file, so that someone with no
# usable runs is listed with 0 instead of being left out.
_all_subs = sorted({ids[0] for ids in
                    (parse_bids_ids(os.path.basename(f)) for f in fix_files) if ids})
_n_runs = (df_trial.drop_duplicates(subset=['sub_id', 'block', 'film_id'])
           .groupby('sub_id').size()
           .reindex(_all_subs, fill_value=0))
_short = _n_runs[_n_runs < N_RUNS_PER_SUBJECT]
if _short.empty:
    print(f"  All {len(_n_runs)} participants have {N_RUNS_PER_SUBJECT} runs.")
else:
    print(f"  WARNING: {len(_short)} participant(s) have fewer than "
          f"{N_RUNS_PER_SUBJECT} usable runs (some may be excluded below):")
    for _sub, _k in _short.items():
        print(f"    sub-{int(_sub):03d}: {_k} runs")

df_trial = df_trial[~df_trial['sub_id'].isin(EXCLUDE_SUBJECTS)]
print(f"  {len(df_trial):,} film-viewing trials across {df_trial['sub_id'].nunique()} subjects "
      f"(after excluding {EXCLUDE_SUBJECTS})")

# ==================================================================
# 2. Group lookup (corrected conditions.csv, same as encoding_bias.py)
# ==================================================================
lookup = load_behav_data(PATH_BEH, PATH_COND).rename(columns={'Participant_ID': 'sub_id'})
for c in ['sub_id', 'session', 'block', 'film_id']:
    lookup[c] = lookup[c].astype(int)

def _assign_group_enc(timing, job):
    if timing == 'Pro' and job == 'Spy':      return 'Spy_Pro'
    if timing == 'Pro' and job == 'Criminal': return 'Crim_Pro'
    return 'Control'

lookup['Group_Enc'] = lookup.apply(lambda r: _assign_group_enc(r['timing'], r['job']), axis=1)

df_trial = df_trial.merge(
    lookup[['sub_id', 'session', 'block', 'film_id', 'Group_Enc']],
    on=['sub_id', 'session', 'block', 'film_id'], how='left')
df_trial = df_trial.dropna(subset=['Group_Enc'])

print(f"  {len(df_trial):,} trials with a group assignment")
print(df_trial['Group_Enc'].value_counts())

# ==================================================================
# 3. Subject-level aggregation + group stats
# ==================================================================
METRICS = ['n_fix', 'mean_fix_dur_ms', 'scanpath_px', 'dispersion_px']
PLOT_METRICS = ['n_fix', 'mean_fix_dur_ms', 'scanpath_px']   # dispersion: modelled, not plotted

subj = df_trial.groupby(['sub_id', 'Group_Enc'])[METRICS].mean().reset_index()

print("\n=== Subject-level group means +/- 95% CI ===")
for metric in METRICS:
    print(f"\n--- {metric} ---")
    for grp in GROUP_ORDER:
        vals = subj.loc[subj['Group_Enc'] == grp, metric].values
        n = len(vals)
        m = vals.mean()
        se = stats.sem(vals)
        half = se * stats.t.ppf(0.975, n - 1)
        print(f"  {grp:<10} n={n:2d}  mean={m:8.1f}  95% CI [{m-half:8.1f}, {m+half:8.1f}]")

# ==================================================================
# 3B. Group significance test - linear mixed model (mirrors
#     encoding_bias.py's Section 6)
# ------------------------------------------------------------------
# Per metric: lmer(metric ~ Group_Enc * block_fac + film_id_fac + (1|sub_id)
#   + (1|sub_id:block_fac) + (1|sub_id:film_id_fac)), REML, via R
# (lme4/lmerTest/emmeans); see the header for the rationale. Reported:
# ANOVA, Tukey-adjusted pairwise group contrasts (+ uncorrected + 95% CI),
# and standardized effect sizes (Cohen's d, standardized by total SD).
# ==================================================================
print("\n" + "="*60)
print("SECTION 3B - Group Significance Test (Linear Mixed Model)")
print("="*60)

df_trial['film_id_fac'] = df_trial['film_id'].astype(str)
df_trial['block_fac']   = df_trial['block'].astype(str)

_R_MODEL_TEMPLATE = r"""
suppressPackageStartupMessages({{
  library(lme4); library(lmerTest); library(emmeans)
}})
options(warn=-1)
emm_options(lmer.df="satterthwaite")
fit_check <- function(m, name) {{
  msgs <- m@optinfo$conv$lme4$messages
  cat(sprintf("%s: isSingular = %s | convergence: %s\n", name, isSingular(m),
              if (length(msgs)) paste(msgs, collapse="; ") else "OK"))
}}
df <- read.csv("{csv_path}")
df$Group_Enc  <- factor(df$Group_Enc,  levels=c("Control","Spy_Pro","Crim_Pro"))
df$block_fac  <- factor(df$block_fac,  levels=c("1","2","3"))
df$sub_id     <- factor(df$sub_id)
df$film_id_fac <- factor(df$film_id_fac)

model <- lmer({dv} ~ Group_Enc * block_fac + film_id_fac + (1|sub_id)
              + (1|sub_id:block_fac) + (1|sub_id:film_id_fac),
              data=df, REML=TRUE)
cat("\n=== Model ({label}) ===\n")
print(formula(model))
cat("\n=== Singular fit / convergence check ({label}) ===\n")
fit_check(model, "main model")
cat("\n=== ANOVA ({label}) ===\n")
print(anova(model))

emm <- emmeans(model, ~ Group_Enc)
cat("\n=== Contrasts - Tukey adjusted ({label}) ===\n")
print(contrast(emm, method="pairwise", adjust="tukey"))
# Same Tukey contrasts saved for the significance brackets in the plot
write.csv(as.data.frame(contrast(emm, method="pairwise", adjust="tukey")),
          "{out_csv}", row.names=FALSE)

cat("\n=== Contrasts - Uncorrected ({label}) ===\n")
print(contrast(emm, method="pairwise", adjust="none"))

cat("\n=== Contrasts - Tukey adjusted, 95% CI ({label}) ===\n")
print(confint(contrast(emm, method="pairwise", adjust="tukey")))

cat("\n=== Standardized effect sizes for group contrasts (Cohen's d) ({label}) ===\n")
cat("d standardized by TOTAL SD = sqrt(random-effect variances + residual + film-effect variance).\n")
vc_es <- as.data.frame(VarCorr(model))
# Film is fixed, so its variance is the (population) variance of the 6 film means;
# adding it keeps d on the same scale as when film was a random effect.
film_means <- summary(emmeans(model, ~ film_id_fac))$emmean
film_var <- mean((film_means - mean(film_means))^2)
cat(sprintf("Film-effect variance (fixed film): %.5f\n", film_var))
total_sd <- sqrt(sum(vc_es$vcov[is.na(vc_es$var2)]) + film_var)
cat(sprintf("Total SD used for standardization: %.4f\n", total_sd))
print(eff_size(emm, sigma=total_sd, edf=df.residual(model)))

cat("\n=== Descriptives ({label}) ===\n")
print(summary(emm))

cat("\n=== Block-wise Contrasts - Tukey adjusted ({label}) ===\n")
emm_block <- emmeans(model, ~ Group_Enc | block_fac)
print(contrast(emm_block, method="pairwise", adjust="tukey"))

cat("\n=== Block-wise Contrasts - Uncorrected ({label}) ===\n")
print(contrast(emm_block, method="pairwise", adjust="none"))

cat("\n=== Block-wise standardized effect sizes for group contrasts (Cohen's d) ({label}) ===\n")
cat("Same standardization as the overall effect size above (total SD = sqrt(random-effect\n")
cat("variances + residual + film-effect variance)), computed separately within each block.\n")
print(eff_size(emm_block, sigma=total_sd, edf=df.residual(model)))

cat("\n=== Block-wise Descriptives ({label}) ===\n")
print(summary(emm_block))

cat("\n=== Random Effect Variance ({label}) ===\n")
print(VarCorr(model))
"""


def _contrast_csv(dv_col):
    """Tukey group contrasts (averaged over blocks) for one measure, written by the R model."""
    return os.path.join(PATH_OUT, f'ET_encoding_search_breadth_contrasts_{dv_col}.csv')


def _run_r_model(data: pd.DataFrame, dv_col: str, label: str):
    """Print descriptives, then fit the mixed model in R and print its output."""
    print(f"\n--- {label} ---")

    desc = (data.groupby('Group_Enc', observed=True)[dv_col]
            .agg(count='count', mean='mean', std='std',
                 min='min', max='max', median='median'))
    print(desc.round(2))

    if not HAS_R:
        print(f"  [Model skipped - Rscript not found at {RSCRIPT}]")
        return
    if data[dv_col].notna().sum() < 20:
        print("  [Model skipped - insufficient data]")
        return

    out_csv = _contrast_csv(dv_col)
    if os.path.exists(out_csv):
        os.unlink(out_csv)   # never let the plot pick up a stale file from an earlier run
    result = run_r(_R_MODEL_TEMPLATE, data, RSCRIPT, dv=dv_col, label=label,
                   out_csv=out_csv.replace(chr(92), '/'))
    print_r_output(result)

for metric in METRICS:
    _run_r_model(df_trial, metric, metric)

# ==================================================================
# 4. Plots
# ==================================================================
# Plot style matched to the bias-score figures in encoding_bias.py
# (ggplot style): serif font, no grid/spines, half-violin with IQR
# lines, jittered dots, black mean point; sized for a full A4 text width.
CENTRE_STAT  = 'mean'   # statistic for the black point: 'mean' matches the reported M values ('median' = plots)
GROUP_LABELS = {'Control': 'Control', 'Spy_Pro': 'Spy', 'Crim_Pro': 'Criminal'}
FIG_WIDTH_IN = 6.7
PAPER_RC = {
    'font.family':     'serif',
    'font.size':       7,
    'axes.titlesize':  8,
    'axes.labelsize':  7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 6,
}


# Significance brackets (same convention as the bias plots): Tukey group
# contrasts averaged over blocks, read from the CSV the R model wrote above;
# *** < .001, ** < .01, * < .05, + < .10 (trend, if SHOW_TRENDS).
SHOW_TRENDS = True


def _draw_brackets(ax, metric):
    """Brackets for significant group contrasts on one panel."""
    path = _contrast_csv(metric)
    if not os.path.exists(path):
        return
    pairs = contrast_pairs(pd.read_csv(path), {g: i for i, g in enumerate(GROUP_ORDER)},
                           SHOW_TRENDS)
    draw_brackets([(ax, pairs)], PAPER_RC['font.size'] + 1)


# (title, scale divisor) - scanpath shown in thousands of px to keep tick labels short
titles = {
    'n_fix':           ('Fixations per viewing', 1),
    'mean_fix_dur_ms': ('Fixation duration (ms)', 1),
    'scanpath_px':     ('Scanpath length (10³ px)', 1000),
}
with plt.rc_context(PAPER_RC):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.8))
    for ax, metric in zip(axes, PLOT_METRICS):
        title, scale = titles[metric]
        for xi, grp in enumerate(GROUP_ORDER):
            vals = subj.loc[subj['Group_Enc'] == grp, metric].values / scale
            raincloud(ax, vals, xi, GROUP_PALETTE[grp], point_size=14)
            ax.scatter([xi], [getattr(np, f'nan{CENTRE_STAT}')(vals)],
                       color='black', marker='o', s=40, zorder=6)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
        ax.set_title(title)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0, colors='black')
        _draw_brackets(ax, metric)

    fig.tight_layout()
    out_path = os.path.join(PATH_PAPER, 'ET_encoding_search_breadth.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
print(f"\nSaved: {out_path}")

_log.close()
