# ==================================================================
# ENCODING PHASE - EYE TRACKING BIAS ANALYSIS
# ==================================================================
#
# Reproduces the encoding eye-tracking statistics and figures of the paper.
# Helper functions (logging, file loading, running R, plotting) are in
# helper.py.
#
# STUDY CONTEXT
# -------------
# Participants watched six films in each of three blocks. Each participant
# was assigned a job role (Spy, Criminal, Neutral) and an instruction timing
# (Pro = told before the film, Retro = told after, Neutral = no role).
# Groups:
#   Spy_Pro    - told "you are a spy" BEFORE watching
#   Crim_Pro   - told "you are a criminal" BEFORE watching
#   Control    - all other conditions (Retro + Neutral), collapsed
#
# DEPENDENT VARIABLES
# -------------------
# Three gaze measures, each expressed as face proportion =
# face / (face + place), range 0–1 (objects excluded from the denominator):
#   fixation_count    - n fixations on face regions / (face + place)
#   fixation_duration - ms dwell on face regions / (face + place)
#   sample_proportion - n raw gaze samples (500 Hz) on face / (face + place)
# Each is converted to a bias score (2 * face_prop - 1), from -1 (only
# places) to +1 (only faces).
#
# STRUCTURE
# ---------
# Section 1   Load BIDS-format ET derivatives + condition labels
# Section 1B  Out-of-bounds (calibration) exclusion gate
# Section 2   Reliability gate (minimum valid fixations/samples)
# Section 4   Build bias score dataset, assign groups
# Section 5   Plot settings and the contrasts drawn as brackets
# Section 5B  Bias score figure, faceted by condition (paper figure)
# Section 5C  Spy/Crim vs per-block Control figure (paper figure)
# Section 6   Mixed models: Bias ~ Group * Block + film + random effects
# Section 7   Retro groups vs Control
# Section 10  Participant-level bias scores for the recall correlations
# Section 11  Category viewing-time breakdown, Friedman tests, radar figure
#
# STATISTICAL MODEL
# -----------------
# lmer(Bias_Score ~ Group_Enc * block_fac + film + (1|Participant_ID)
#      + (1|Participant_ID:block_fac) + (1|Participant_ID:film))
# - Group is between-subjects; block and film are within-subjects and
#   fully crossed (each participant sees all 6 films in each of 3 blocks).
# - (1|Participant_ID:block_fac) is the non-singular simplification of a
#   random block slope; (1|Participant_ID:film) accounts for seeing the
#   same film 3 times; film is fixed because there are only 6 films.
# - Post-hoc pairwise contrasts with Tukey correction.
# - REML fit, Satterthwaite df. Sensitivity models are printed in Section 6.
#
# Input:   paths set in config.py: BIDS derivatives (fixation and eye1 sample
#          files with .json sidecars), shamem_encoding_behav.csv, conditions.csv
# Output:  <RESULTS_DIR>/ paper figures
#          <RESULTS_DIR>/eyetracking_results/ logs, contrast tables, participant-level
#          exports, long-format dataset
# ==================================================================

import os
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats as _sp_stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config
from helper import (Tee, load_et_data, load_behav_data, run_r, print_r_output,
                    raincloud, centre_point, style_ax, sig_pairs, draw_brackets,
                    polar_interp)

# Path to Rscript - R needs lme4, lmerTest and emmeans installed
RSCRIPT = config.RSCRIPT
HAS_R = shutil.which(RSCRIPT) is not None

# ── Paths ──────────────────────────────────────────────────────────

PATH_ET   = config.ET_DERIVATIVES
PATH_BEH  = config.BEHAV_ENCODING_CSV
PATH_COND = config.CONDITIONS_CSV
# Paper figures go to RESULTS_DIR; everything else (logs, CSVs, figures for
# the other two measures) goes to RESULTS_DIR/eyetracking_results/.
PATH_PAPER = config.RESULTS_DIR
PATH_OUT   = os.path.join(PATH_PAPER, "eyetracking_results")
PAPER_FIGURES = {
    'ET_encoding_bias_by_condition_fixation_count.png',
    'ET_encoding_bias_vs_control_fixation_count.png',
    'ET_category_breakdown_radar.png',
}


def _fig_path(name):
    """Save path for a figure: RESULTS_DIR if it is a paper figure, else eyetracking_results/."""
    return os.path.join(PATH_PAPER if name in PAPER_FIGURES else PATH_OUT, name)

# Participants excluded from all analyses.
# Two different bases, unioned into EXCLUDE_SUBJECTS after ET data loads
# (Section 1B) so the automated part can't silently go stale:
#  - MANUAL_EXCLUDE_SUBJECTS: cases that can't be caught by an automated
#    data-quality rule below, documented by hand.
#  - QC_EXCLUDE_SUBJECTS (Section 1B): high OutOfBounds (OOB) rate in the
#    BIDS derivatives, a symptom of gaze-tracking calibration drift.
#    Recomputed fresh from the data on every run. This is expected to catch
#    subs 7, 25, 48, 52, 58, 84 (all >=10% OOB - see Section 1B printout).
MANUAL_EXCLUDE_SUBJECTS = [
    16, 68, 74, 94, 95,
    85,  # fixation coverage collapsed mid-session (fatigue)
    # 59, 79, 90: same top-shifted gaze / calibration-drift artifact as the
    # OOB-flagged participants (raw gaze mean Y 110-230px above clean
    # participants; session notes record five-point recalibration for all
    # three), but their OOB rate stays under the
    # Section 1B threshold because the offset happens to keep gaze on-screen.
    # No calibration-validation or drift-correction records survive in the
    # raw .asc exports to measure/reverse the offset directly, so these stay
    # a documented manual exclusion rather than an automated one.
    59, 79, 90,
]
os.makedirs(PATH_OUT, exist_ok=True)

MEASURE_NAMES = ['fixation_count', 'fixation_duration', 'sample_proportion']

# Everything printed goes to the console and to log_encoding_ET_bias.txt;
# per-measure model output goes to log_encoding_ET_bias_<measure>.txt.
_log = Tee(os.path.join(PATH_OUT, 'log_encoding_ET_bias.txt'), MEASURE_NAMES, PATH_OUT)
sys.stdout = _log

# Colour palette (colour-blind friendly)
GROUP_PALETTE = {
    'Control':  '#A6A6A6',
    'Spy_Pro':  '#0072B2',
    'Crim_Pro': '#D55E00',
}


# ==================================================================
# SECTION 1 - LOAD & PREPARE DATA
# ------------------------------------------------------------------
# Two data sources are merged on (Participant_ID, session, block, film):
#
# 1. BIDS-format ET derivatives (one file per run = film x block, ses-01):
#    - Fixation physioevents (*_recording-fixations_physioevents.tsv.gz):
#      each row = one fixation event, with duration (seconds),
#      Category_Start (which semantic region the fixation landed on),
#      and OutOfBounds flag. Aggregated to face/place counts + durations.
#    - Sample physio (*_recording-eye1_physio.tsv.gz): raw gaze samples,
#      same category tags. Kept as a second convergent measure.
#      Only eye1 is read - eye1/eye2 carry identical category columns
#      (one pixel per sample, left eye preferred), so reading both
#      would double count.
#    Files are header-less; column names come from the JSON sidecars.
#    Film and block are taken from the data columns (Movie /
#    source_chunk_key and block), not from the filename.
#
# 2. Behavioural CSV (shamem_encoding_behav.csv):
#    Contains job and timing codes per film viewing. Collapsed to one
#    row per (subject × session × block × film) since these labels are
#    constant within a film - the many trial-level rows are redundant.
#
# Result: df_long - one row per (subject, film, block, measure),
# DV_Value = face proportion for that measure.
# ==================================================================

# ── Execute loading ────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1 - Load & Prepare Data")
print("="*60)

df_et = load_et_data(PATH_ET)

print("\n  Loading behavioural data (corrected conditions from conditions.csv)…")
beh_lookup = load_behav_data(PATH_BEH, PATH_COND)

# Align key types before merge
for col in ['Participant_ID', 'session', 'block', 'film_id']:
    beh_lookup[col] = beh_lookup[col].astype(int)

df_et['Participant_ID'] = df_et['sub_id'].astype(int)
df_et['session']        = df_et['session'].astype(int)
df_et['block']          = df_et['block'].astype(int)
df_et['film_id']        = df_et['film_id'].astype(int)


# ==================================================================
# SECTION 1B - CALIBRATION/TRACKING-QUALITY GATE (OOB rate)
# ------------------------------------------------------------------
# The bias score is derived directly from gaze -> semantic-category
# assignment, with no per-participant calibration correction anywhere
# in the pipeline. A high OutOfBounds (OOB) rate - gaze mapped outside
# the valid screen area - is the pipeline's own validity flag and
# signals a miscalibrated/drifted eye tracker for that participant,
# not a real face/place preference (spot-checked: sub-007's raw gaze
# has a mean Y ~200px *above* the top screen edge; clean participants
# sit around Y=450-460 on a 1920x1080 screen).
#
# Participants with a mean OOB rate >= 10% in fixation OR sample files
# are auto-excluded. The 10% threshold was chosen against a ~0.1-0.2%
# baseline observed in clean participants. This is computed fresh from
# the BIDS derivatives on every run, so it can't go stale like a
# hand-maintained list would.
#
# Some participants (59, 79, 90 - see MANUAL_EXCLUDE_SUBJECTS above)
# show the same calibration-drift signature but a smaller offset that
# keeps gaze on-screen, so they stay under this threshold and are
# excluded manually instead.
# ==================================================================
print("\n" + "="*60)
print("SECTION 1B - Calibration/Tracking-Quality Gate (OOB rate)")
print("="*60)

OOB_RATE_THRESHOLD = 0.10

oob_participant = (
    df_et.groupby('Participant_ID')[
        ['fix_rows_total', 'fix_rows_oob', 'samp_rows_total', 'samp_rows_oob']
    ]
    .sum()
    .reset_index()
)
oob_participant['fixation_oob_rate'] = (
    oob_participant['fix_rows_oob'] / oob_participant['fix_rows_total'])
oob_participant['sample_oob_rate'] = (
    oob_participant['samp_rows_oob'] / oob_participant['samp_rows_total'])
oob_participant['high_oob'] = (
    (oob_participant['fixation_oob_rate'] >= OOB_RATE_THRESHOLD) |
    (oob_participant['sample_oob_rate']   >= OOB_RATE_THRESHOLD)
)

QC_EXCLUDE_SUBJECTS = sorted(
    oob_participant.loc[oob_participant['high_oob'], 'Participant_ID'].tolist())

print(f"  OOB rate threshold: >= {OOB_RATE_THRESHOLD:.0%} (fixation or sample)")
print(f"  Flagged participants: {QC_EXCLUDE_SUBJECTS}")
if QC_EXCLUDE_SUBJECTS:
    print(oob_participant.loc[oob_participant['high_oob'],
                              ['Participant_ID', 'fixation_oob_rate', 'sample_oob_rate']]
          .round(3).sort_values('sample_oob_rate', ascending=False)
          .to_string(index=False))

EXCLUDE_SUBJECTS = sorted(set(MANUAL_EXCLUDE_SUBJECTS) | set(QC_EXCLUDE_SUBJECTS))
print(f"\n  Manual exclusions:     {MANUAL_EXCLUDE_SUBJECTS}")
print(f"  QC-flagged exclusions: {QC_EXCLUDE_SUBJECTS}")
print(f"  Combined EXCLUDE_SUBJECTS: {EXCLUDE_SUBJECTS}")


df_wide = df_et.merge(
    beh_lookup,
    on=['Participant_ID', 'session', 'block', 'film_id'],
    how='left',
)
df_wide = df_wide[~df_wide['Participant_ID'].isin(EXCLUDE_SUBJECTS)]
print(f"\n  Excluded subjects: {EXCLUDE_SUBJECTS}")

print(f"\n  Merged wide dataset: {len(df_wide)} rows | "
      f"{df_wide['Participant_ID'].nunique()} subjects, "
      f"{df_wide['block'].nunique()} blocks, "
      f"{df_wide['film_id'].nunique()} films")
print(df_wide.describe())

# n_valid: total face+place fixations/samples backing each proportion.
# fixation_count and fixation_duration share the same underlying fixations
# (duration is just those same fixations' dwell time), so both use the
# fixation total; sample_proportion uses the 500 Hz sample total instead.
# Carried through the melt so Section 2 can gate on reliability (how much
# data backs a proportion) rather than on the proportion's value itself.
df_wide['n_valid_fixation_count']    = (df_wide['face_fix_count'].fillna(0)
                                        + df_wide['place_fix_count'].fillna(0))
df_wide['n_valid_fixation_duration'] = df_wide['n_valid_fixation_count']
df_wide['n_valid_sample_proportion'] = (df_wide['face_sample_count'].fillna(0)
                                        + df_wide['place_sample_count'].fillna(0))

# Melt to long format: one row per (subject, film, block, measure)
id_cols = ['Participant_ID', 'session', 'block', 'film_id', 'job', 'timing']
df_long = df_wide.melt(
    id_vars=id_cols,
    value_vars=['face_count_prop', 'face_duration_prop', 'face_sample_prop'],
    var_name='measure',
    value_name='DV_Value',
)
df_long['measure'] = df_long['measure'].map({
    'face_count_prop':    'fixation_count',
    'face_duration_prop': 'fixation_duration',
    'face_sample_prop':   'sample_proportion',
})

n_valid_long = df_wide.melt(
    id_vars=id_cols,
    value_vars=['n_valid_fixation_count', 'n_valid_fixation_duration',
                'n_valid_sample_proportion'],
    var_name='measure',
    value_name='N_Valid',
)
n_valid_long['measure'] = n_valid_long['measure'].str.replace('n_valid_', '', regex=False)

df_long = df_long.merge(n_valid_long, on=id_cols + ['measure'], how='left')
df_long = df_long.dropna(subset=['DV_Value', 'job', 'timing'])

print(f"\n  Long-format dataset: {len(df_long)} rows")


# ==================================================================
# SECTION 2 - RELIABILITY GATE (minimum valid fixations/samples)
# ------------------------------------------------------------------
# Previously this step flagged observations as outliers using a 3 SD
# cross-participant z-score on the face proportion itself. That rule
# was dropped: because face_prop is bounded [0,1] and most conditions
# cluster high, mean+3SD routinely exceeded the ceiling (nobody could
# ever be flagged as "too face-biased") while mean-3SD stayed in-range
# (anyone genuinely place-biased got flagged) - a mechanically
# one-directional rule. Checked on real data: ALL 176 removed
# observations in the last SD-based run were on the "more place" side,
# in a condition (Crim_Pro) specifically hypothesised to contain real
# place-biased responders. That rule risked deleting the effect itself.
#
# Replacement: exclude an observation only if too little raw data backs
# its proportion to trust it - a criterion orthogonal to whether the
# result is face- or place-biased, so it can't selectively eat one
# direction of the hypothesised effect.
#   fixation_count / fixation_duration - need >= MIN_VALID_FIX total
#     face+place fixations (both measures share the same fixations;
#     duration is just those same events' dwell time).
#   sample_proportion - need >= MIN_VALID_SAMPLE total face+place
#     samples (500 Hz gaze stream).
#
# Note: OutOfBounds fixations/samples are already excluded inside the
# loading functions (_aggregate_fixation, _aggregate_sample) before
# N_Valid is even computed - this gate is about too little *valid*
# data, not invalid data (that's handled upstream).
# ==================================================================
print("\n" + "="*60)
print("SECTION 2 - Reliability Gate (minimum valid fixations/samples)")
print("="*60)

MIN_VALID_FIX    = 5    # min total face+place fixations (fixation_count/duration)
MIN_VALID_SAMPLE = 50   # min total face+place samples (sample_proportion, 500 Hz)

_MIN_VALID_BY_MEASURE = {
    'fixation_count':    MIN_VALID_FIX,
    'fixation_duration': MIN_VALID_FIX,
    'sample_proportion': MIN_VALID_SAMPLE,
}

df_clean = df_long.copy()
df_clean['Min_Required'] = df_clean['measure'].map(_MIN_VALID_BY_MEASURE)
df_clean['Keep']         = df_clean['N_Valid'] >= df_clean['Min_Required']

n_total   = len(df_clean)
n_removed = (~df_clean['Keep']).sum()
print(f"  Observations removed: {n_removed} ({n_removed/n_total*100:.2f}%)")
print(f"  Thresholds - fixation_count/duration: >= {MIN_VALID_FIX} fixations | "
      f"sample_proportion: >= {MIN_VALID_SAMPLE} samples")

print("\n  --- N_Valid distribution per measure (all observations) ---")
print(df_clean.groupby('measure', observed=True)['N_Valid']
      .describe(percentiles=[.01, .05, .1, .25, .5]).round(1))

print("\n  --- Removed observations by condition (job x timing x measure) ---")
removed = df_clean[~df_clean['Keep']]
if len(removed) > 0:
    print(removed.groupby(['job', 'timing', 'measure'], observed=True).size()
          .unstack(fill_value=0))
else:
    print("  None removed at current thresholds.")

print("\n  --- FOCUS: Criminal.Pro (hypothesised place-bias responders) ---")
cp_mask = (df_clean['job'] == 'Criminal') & (df_clean['timing'] == 'Pro')
cp_removed = df_clean[cp_mask & ~df_clean['Keep']]
if len(cp_removed) > 0:
    print(f"  Removed: {len(cp_removed)} obs (too few valid fixations/samples):")
    print(cp_removed[['block', 'measure', 'DV_Value', 'N_Valid']]
          .sort_values('N_Valid').to_string(index=False))
else:
    print("  No Criminal.Pro observations removed.")


# ==================================================================
# SECTION 4 - BUILD BIAS SCORE DATASET
# ------------------------------------------------------------------
# Group assignment:
#   Spy_Pro    = job Spy    + timing Pro  (prospective spy instruction)
#   Crim_Pro   = job Criminal + timing Pro  (prospective criminal instruction)
#   Control    = everything else (Retro + Neutral), collapsed into one
#                baseline group. The sanity check (Section 7) validates
#                that Retro Spy and Retro Criminal do not differ from
#                each other or from Neutral, justifying this collapse.
#
# Bias score formula:
#   Bias = 2 * face_prop - 1    (range -1 to +1)
#   This is mathematically equivalent to (face - place) / (face + place),
#   which is the same as R's Face_Val - Place_Val on the proportion scale.
#   Positive = more fixations/time on faces than places.
#   Zero = equal attention to faces and places.
#   Negative = more fixations/time on places than faces.
# ==================================================================
print("\n" + "="*60)
print("SECTION 4 - Build Bias Score Dataset")
print("="*60)


def _assign_group_enc(timing: str, job: str) -> str:
    if timing == 'Pro' and job == 'Spy':      return 'Spy_Pro'
    if timing == 'Pro' and job == 'Criminal': return 'Crim_Pro'
    return 'Control'


df_enc = df_clean[df_clean['Keep']].copy()
df_enc['Group_Enc'] = df_enc.apply(
    lambda r: _assign_group_enc(r['timing'], r['job']), axis=1)
df_enc['Group_Enc'] = pd.Categorical(
    df_enc['Group_Enc'], categories=['Control', 'Spy_Pro', 'Crim_Pro'])

# Bias score: face_prop – place_prop = 2*face_prop – 1  (range –1 to +1)
# Equivalent to R's  Face_Val – Place_Val  (positive = preference for faces)
df_enc['Bias_Score'] = 2.0 * df_enc['DV_Value'] - 1.0

df_enc = df_enc.rename(columns={'film_id': 'film'})
df_enc['film']      = df_enc['film'].astype(str)
df_enc['block_fac'] = df_enc['block'].astype(str)

print(f"  Retained: {len(df_enc)} observations after outlier removal")
print(df_enc[['Group_Enc', 'block', 'measure']].value_counts().sort_index())

# Long-format export (one row per subject/film/block/measure): the data the
# Section 6 models are fitted to.
df_enc.to_csv(os.path.join(PATH_OUT, 'ET_encoding_bias_long.csv'), index=False)


# ==================================================================
# SECTION 5 - PLOT SETTINGS AND BRACKET CONTRASTS
# ------------------------------------------------------------------
# Shared settings for the bias-score figures (Sections 5B and 5C), and the
# Tukey contrasts those figures draw as significance brackets.
# ==================================================================
print("\n" + "="*60)
print("SECTION 5 - Plot settings and bracket contrasts")
print("="*60)

# Plot style matched to the ggplot bias-score figures:
# serif font, no grid/spines, half-violin with IQR lines, jittered dots,
# black mean point. Sizes are set for a full A4 text-width figure
# (~6.7 in wide), so fonts print at roughly 9-10 pt after LaTeX scaling.
CENTRE_STAT = 'mean'   # statistic for the black points/line: 'mean' matches the reported M values ('median' = plots)
GROUP_LABELS = {'Control': 'Control', 'Spy_Pro': 'Spy', 'Crim_Pro': 'Criminal'}
BIAS_YLABEL  = 'bias score (space < 0 < person)'
FIG_WIDTH_IN = 6.7
PAPER_RC = {
    'font.family':     'serif',
    'font.size':       7,
    'axes.titlesize':  8,
    'axes.labelsize':  7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
}

# Column keeps the name Mean_Bias for brevity; it holds CENTRE_STAT
means_enc = (df_enc.groupby(['block', 'Group_Enc', 'measure'], observed=True)['Bias_Score']
             .agg(CENTRE_STAT).reset_index().rename(columns={'Bias_Score': 'Mean_Bias'}))

measures = sorted(df_enc['measure'].unique())
blocks   = sorted(df_enc['block'].unique())
nrows, ncols = len(measures), len(blocks)

group_order = list(GROUP_PALETTE.keys())


# ------------------------------------------------------------------
# Significance brackets for the bias plots
# The Section 6 model (Group_Enc * block_fac + _R_RE_MAIN, see below)
# is fitted here once per measure, before plotting, and its Tukey-adjusted
# contrasts are saved to RESULTS_DIR/eyetracking_results/ET_encoding_bias_contrasts_<measure>.csv:
#   family 'group_by_block' - group pairs within each block
#   family 'block_by_group' - block pairs within each group
# The plots draw a bracket for every contrast with p < SIG_MAX_P, labelled
# *** < .001, ** < .01, * < .05, + < .10 (trend, if SHOW_TRENDS).
# ------------------------------------------------------------------
SHOW_TRENDS = True
SIG_MAX_P   = 0.10 if SHOW_TRENDS else 0.05
BRACKET_FONTSIZE = PAPER_RC['font.size'] + 1

# ------------------------------------------------------------------
# Random-effects structure shared by every Section 6 model (main stats,
# plot brackets) so they cannot drift apart.
#
# Design: Group is BETWEEN-subjects (one group per participant); block
# (1-3) and film (1-6) are WITHIN-subjects and fully crossed - every
# participant sees all 6 films in every block, i.e. each film 3 times.
#   (1|Participant_ID)           - baseline face preference per person
#   (1|Participant_ID:block_fac) - person-specific deviation per block
#                                  (random block effect per participant;
#                                  needed because block is within-subject
#                                  and Group x block is the key test)
#   (1|Participant_ID:film)      - person-specific response to a given film,
#                                  repeated across the 3 viewings
#   film (FIXED effect)          - film content (more face vs place); fixed
#                                  rather than (1|film) because there are
#                                  only 6 films.
# The maximal model (1 + block_fac|Participant_ID) is fitted as a
# sensitivity check in Section 6; it is singular on these data (block-2
# and block-3 slopes correlate ~.99), so (1|Participant_ID:block_fac) is
# used as its non-singular, compound-symmetric simplification.
# _R_RE_MAIN is appended after the design terms ("Group_Enc * block_fac + ").
# ------------------------------------------------------------------
_R_RE_MAIN = ("film + (1|Participant_ID) + (1|Participant_ID:block_fac) "
              "+ (1|Participant_ID:film)")

_R_PLOT_CONTRASTS = r"""
suppressPackageStartupMessages({{
  library(lme4); library(lmerTest); library(emmeans)
}})
options(warn=-1)
emm_options(lmer.df="satterthwaite")
df <- read.csv("{csv_path}")
df$Group_Enc  <- factor(df$Group_Enc,  levels=c("Control","Spy_Pro","Crim_Pro"))
df$block_fac  <- factor(df$block_fac,  levels=c("1","2","3"))
df$Participant_ID <- factor(df$Participant_ID)
df$film       <- factor(df$film)
model <- lmer(Bias_Score ~ Group_Enc * block_fac + {re},
              data=df, REML=TRUE)
a <- as.data.frame(contrast(emmeans(model, ~ Group_Enc | block_fac),
                            method="pairwise", adjust="tukey"))
a$family <- "group_by_block"; a$by <- as.character(a$block_fac)
b <- as.data.frame(contrast(emmeans(model, ~ block_fac | Group_Enc),
                            method="pairwise", adjust="tukey"))
b$family <- "block_by_group"; b$by <- as.character(b$Group_Enc)
b$contrast <- gsub("block_fac", "", b$contrast)
cols <- c("family", "by", "contrast", "estimate", "SE", "df", "t.ratio", "p.value")
write.csv(rbind(a[cols], b[cols]), "{out_csv}", row.names=FALSE)
"""


def _plot_contrasts(data, meas):
    """Fit the Section 6 model for one measure and return its Tukey contrasts
    (see _R_PLOT_CONTRASTS), also saved to eyetracking_results/. Empty frame if R is missing."""
    empty = pd.DataFrame(columns=['family', 'by', 'contrast', 'estimate', 'p.value'])
    if not HAS_R or data['Bias_Score'].notna().sum() < 20:
        return empty
    out_csv = os.path.join(PATH_OUT, f'ET_encoding_bias_contrasts_{meas}.csv')
    if os.path.exists(out_csv):
        os.unlink(out_csv)   # never fall back to a stale file from an earlier run
    result = run_r(_R_PLOT_CONTRASTS, data, RSCRIPT,
                   out_csv=out_csv.replace('\\', '/'), re=_R_RE_MAIN)
    if not os.path.exists(out_csv):
        print(f"  [plot contrasts skipped for {meas}] {result.stderr.strip()[-300:]}")
        return empty
    print(f"  Saved: {out_csv}")
    res = pd.read_csv(out_csv, dtype={'by': str})
    res['contrast'] = res['contrast'].str.strip()
    return res


plot_contrasts = {}
for meas in measures:
    plot_contrasts[meas] = _plot_contrasts(df_enc[df_enc['measure'] == meas].copy(), meas)


# ==================================================================
# SECTION 5B - BIAS SCORE PLOTS, FACETED BY CONDITION (progression view)
# ------------------------------------------------------------------
# Rainclouds faceted to make the across-block PROGRESSION within each
# condition the primary reading:
#   - one figure per ET measure
#   - one panel per condition (Control, Spy_Pro, Crim_Pro)
#   - x-axis = block (1-3) within each panel
#   - black points = per-block group means, connected by a line so the
#     block-1 -> block-3 trajectory is visible at a glance
# Solid line at 0 = no bias. Reuses raincloud() and means_enc.
#
# Saved to: RESULTS_DIR/ET_encoding_bias_by_condition_fixation_count.png (paper figure)
#           RESULTS_DIR/eyetracking_results/ (the other two measures)
# ==================================================================
print("\n" + "="*60)
print("SECTION 5B - Bias Score Plot (faceted by condition)")
print("="*60)

for meas in measures:
  with plt.rc_context(PAPER_RC):
    fig, axes_row = plt.subplots(1, len(group_order),
                                  figsize=(FIG_WIDTH_IN, 2.8),
                                  sharey='row', squeeze=False)
    axes_row = axes_row[0]

    for gi, grp in enumerate(group_order):
        ax    = axes_row[gi]
        color = GROUP_PALETTE[grp]
        centre_x, centre_y = [], []
        for xi, blk in enumerate(blocks):
            grp_vals = df_enc[(df_enc['measure'] == meas)
                              & (df_enc['block'] == blk)
                              & (df_enc['Group_Enc'] == grp)]['Bias_Score'].values
            raincloud(ax, grp_vals, xi, color)
            row = means_enc[(means_enc['measure'] == meas)
                            & (means_enc['block'] == blk)
                            & (means_enc['Group_Enc'] == grp)]
            if not row.empty:
                centre_x.append(xi)
                centre_y.append(row['Mean_Bias'].values[0])

        # Connecting line through the per-block means shows the progression
        if centre_x:
            ax.plot(centre_x, centre_y, color='black', linewidth=1.0, zorder=5)
            for x, y in zip(centre_x, centre_y):
                centre_point(ax, x, y)

        style_ax(ax)
        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels([f'Block {b}' for b in blocks])
        ax.set_title(GROUP_LABELS[grp])
        if gi == 0:
            ax.set_ylabel(BIAS_YLABEL)

    draw_brackets([(axes_row[gi], sig_pairs(plot_contrasts[meas], 'block_by_group', grp,
                                            {str(b): i for i, b in enumerate(blocks)},
                                            SHOW_TRENDS))
                   for gi, grp in enumerate(group_order)],
                  BRACKET_FONTSIZE)
    fig.tight_layout()
    out_path = _fig_path(f'ET_encoding_bias_by_condition_{meas}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==================================================================
# SECTION 5C - SPY/CRIM vs PER-BLOCK CONTROL REFERENCE
# ------------------------------------------------------------------
# Uses each block's Control mean as a "neutral" reference line (dotted)
# instead of the fixed 0 line, and plots ONLY Spy_Pro and Crim_Pro in
# contrast to it. Because Control itself can drift across blocks, this
# removes that shared drift so a Spy/Crim deviation is read against the
# actual baseline for that block, not against an absolute zero.
#   - one figure per ET measure
#   - one panel per block (1-3)
#   - Spy_Pro (blue) and Crim_Pro (orange) rainclouds only
#   - dotted horizontal line = that block's Control mean (the reference)
#   - black points = Spy/Crim per-block means
# Expected pattern: Spy_Pro sits increasingly ABOVE the line, Crim_Pro
# increasingly BELOW it, across blocks 1 -> 3.
#
# Values are raw Bias_Score (not re-centred), so the vertical gap between
# a group's diamond and the dotted Control line is the group-vs-Control
# difference for that block, directly readable on the bias-score scale.
#
# Saved to: RESULTS_DIR/ET_encoding_bias_vs_control_fixation_count.png (paper figure)
#           RESULTS_DIR/eyetracking_results/ (the other two measures)
# ==================================================================
print("\n" + "="*60)
print("SECTION 5C - Spy/Crim vs per-block Control reference")
print("="*60)

_FOCUS_GROUPS = ['Spy_Pro', 'Crim_Pro']   # Control is the reference, not plotted

for meas in measures:
  with plt.rc_context(PAPER_RC):
    fig, axes_row = plt.subplots(1, len(blocks),
                                  figsize=(FIG_WIDTH_IN, 2.8),
                                  sharey='row', squeeze=False)
    axes_row = axes_row[0]

    for j, blk in enumerate(blocks):
        ax = axes_row[j]

        # Per-block Control mean = the neutral reference line
        ctrl_row = means_enc[(means_enc['measure'] == meas)
                             & (means_enc['block'] == blk)
                             & (means_enc['Group_Enc'] == 'Control')]
        ctrl_mean = ctrl_row['Mean_Bias'].values[0] if not ctrl_row.empty else np.nan

        for xi, grp in enumerate(_FOCUS_GROUPS):
            color    = GROUP_PALETTE[grp]
            grp_vals = df_enc[(df_enc['measure'] == meas)
                              & (df_enc['block'] == blk)
                              & (df_enc['Group_Enc'] == grp)]['Bias_Score'].values
            raincloud(ax, grp_vals, xi, color)
            row = means_enc[(means_enc['measure'] == meas)
                            & (means_enc['block'] == blk)
                            & (means_enc['Group_Enc'] == grp)]
            if not row.empty:
                centre_point(ax, xi, row['Mean_Bias'].values[0])

        style_ax(ax)
        if not np.isnan(ctrl_mean):
            ax.axhline(ctrl_mean, linestyle=':', color='#555555', linewidth=1.4,
                       zorder=1, label=f'Control {CENTRE_STAT}')
        ax.set_xticks(range(len(_FOCUS_GROUPS)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in _FOCUS_GROUPS])
        ax.set_title(f'Block {blk}')
        if j == 0:
            ax.set_ylabel(BIAS_YLABEL)
        # Label the reference line directly (as in the plot) at the
        # right end of the last panel, clear of the violins and the brackets
        if j == len(blocks) - 1 and not np.isnan(ctrl_mean):
            ax.text(ax.get_xlim()[1], ctrl_mean, ' Control', ha='left', va='center',
                    color='#555555', fontsize=PAPER_RC['legend.fontsize'], clip_on=False)

    draw_brackets([(axes_row[j], sig_pairs(plot_contrasts[meas], 'group_by_block', blk,
                                           {g: i for i, g in enumerate(_FOCUS_GROUPS)},
                                           SHOW_TRENDS))
                   for j, blk in enumerate(blocks)],
                  BRACKET_FONTSIZE)
    fig.tight_layout()
    out_path = _fig_path(f'ET_encoding_bias_vs_control_{meas}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ==================================================================
# SECTION 6 - MIXED MODELS & CONTRASTS
# ------------------------------------------------------------------
# Model (run separately per ET measure):
#   lmer(Bias_Score ~ Group_Enc * block_fac + film
#        + (1|Participant_ID) + (1|Participant_ID:block_fac)
#        + (1|Participant_ID:film))
#
# Fixed effects:
#   Group_Enc     - main effect of instruction group (3 levels, between-subjects)
#   block_fac     - main effect of viewing repetition (blocks 1-3, within)
#   Group × block - interaction: does the bias grow/shrink with repetition?
#   film          - control for film content (some films have more
#                   face/place content by design); 6 levels, fixed.
#                   Group / block emmeans and contrasts average over films.
#
# Random effects (see _R_RE_MAIN for the rationale):
#   (1|Participant_ID)           - baseline face preference varies by person
#   (1|Participant_ID:block_fac) - people differ in how their bias changes
#                                  across blocks. Without it, Group x block
#                                  and block contrasts are tested against
#                                  the trial-level residual (df ~1300 instead
#                                  of ~80-130) and are anti-conservative.
#   (1|Participant_ID:film)      - the same person sees each film 3 times
#
# Sensitivity models (printed after the main output, same Group x block terms):
#   S1 maximal:  film + (1 + block_fac|Participant_ID) + (1|Participant_ID:film)
#                [first proposal; singular on these data]
#   S2 linear:   film + (1 + block_num|Participant_ID) + (1|Participant_ID:film)
#   S3 random film: (1|Participant_ID) + (1|Participant_ID:block_fac)
#                + (1|Participant_ID:film) + (1|film)
#
# Post-hoc contrasts (Tukey-corrected, 3 pairwise tests):
#   Spy_Pro vs Control   - is Spy bias different from baseline?
#   Crim_Pro vs Control  - is Criminal bias different from baseline?
#   Spy_Pro vs Crim_Pro  - is the dissociation significant? (KEY TEST)
# Each group contrast is reported with a 95% CI in bias-score units and a
# standardized effect size (Cohen's d), standardized by the TOTAL SD
# (sqrt of all random-effect variances + residual variance + the variance
# of the 6 film effects) so d is comparable to a conventional
# between-subjects Cohen's d. The film-effect variance is added because film
# is now fixed: without it the SD would shrink purely because film variance
# moved out of the random part, inflating d relative to earlier versions.
#
# Block-wise contrasts repeat the same 3 tests within each block to
# show whether the dissociation is present from the first exposure or
# develops over repetitions.
#
# Block comparisons (new):
#   Marginal block main effect - block1 vs 2, 1 vs 3, 2 vs 3, collapsed
#     across Group_Enc. Answers: does the bias change across viewing
#     repetitions overall?
#   Within-group block contrasts - the same 3 block comparisons repeated
#     separately for Control, Spy_Pro, and Crim_Pro. Follows up the
#     Group x block interaction: does each group's bias change across
#     blocks on its own?
#
# Fitted with REML (the standard for F-tests / contrasts on fixed effects);
# Satterthwaite df throughout. Film is tested by its F-test in the ANOVA
# table (fixed effect), replacing the earlier LRT for (1|film).
# Convergence and singularity are reported for every fit.
# ==================================================================
print("\n" + "="*60)
print("SECTION 6 - Mixed Models & Contrasts")
print("="*60)

# Shared R helper: print singularity + convergence status for a fit.
# (options(warn=-1) silences lme4 warnings, so they must be checked explicitly.)
_R_FIT_CHECK = r"""
fit_check <- function(m, name) {{
  msgs <- m@optinfo$conv$lme4$messages
  cat(sprintf("%s: isSingular = %s | convergence: %s\n", name, isSingular(m),
              if (length(msgs)) paste(msgs, collapse="; ") else "OK"))
}}
"""

_R_MODEL_TEMPLATE = r"""
suppressPackageStartupMessages({{
  library(lme4); library(lmerTest); library(emmeans)
}})
options(warn=-1)
emm_options(lmer.df="satterthwaite")
""" + _R_FIT_CHECK + r"""
df <- read.csv("{csv_path}")
df$Group_Enc  <- factor(df$Group_Enc,  levels=c("Control","Spy_Pro","Crim_Pro"))
df$block_fac  <- factor(df$block_fac,  levels=c("1","2","3"))
df$block_num  <- as.numeric(as.character(df$block_fac)) - 2
df$Participant_ID <- factor(df$Participant_ID)
df$film       <- factor(df$film)

model <- lmer({dv} ~ Group_Enc * block_fac + {re},
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

cat("\n=== Contrasts - Uncorrected ({label}) ===\n")
print(contrast(emm, method="pairwise", adjust="none"))

cat("\n=== Contrasts - Tukey adjusted, 95% CI ({label}) ===\n")
cat("Estimates in bias-score units (range -1 to +1); interpret directly on that scale.\n")
print(confint(contrast(emm, method="pairwise", adjust="tukey")))

cat("\n=== Standardized effect sizes for group contrasts (Cohen's d) ({label}) ===\n")
cat("d standardized by TOTAL SD = sqrt(random-effect variances + residual + film-effect variance),\n")
cat("comparable to a between-subjects Cohen's d. edf = residual df.\n")
vc_es <- as.data.frame(VarCorr(model))
# variances only (var2 is NA); covariance rows would appear if slopes were added.
# Film is fixed, so its variance is the (population) variance of the 6 film means.
film_means <- summary(emmeans(model, ~ film))$emmean
film_var <- mean((film_means - mean(film_means))^2)
cat(sprintf("Film-effect variance (fixed film): %.5f\n", film_var))
total_sd <- sqrt(sum(vc_es$vcov[is.na(vc_es$var2)]) + film_var)
cat(sprintf("Total SD used for standardization: %.4f\n", total_sd))
print(eff_size(emm, sigma=total_sd, edf=df.residual(model)))

cat("\n=== Face-Place split per group (bias vs zero) - Bonferroni ({label}) ===\n")
cat("Tests whether each group's bias score is significantly != 0.\n")
cat("Positive = face preference; Negative = place preference.\n")
print(test(emm, null=0, adjust="bonferroni"))

cat("\n=== Face-Place split per group (bias vs zero) - Uncorrected ({label}) ===\n")
print(test(emm, null=0, adjust="none"))

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

cat("\n=== Block main effect - marginal means, collapsed over Group ({label}) ===\n")
cat("Tests whether the bias score changes across viewing repetitions overall.\n")
emm_block_marg <- emmeans(model, ~ block_fac)
print(summary(emm_block_marg))
cat("\n--- Block 1 vs 2, 1 vs 3, 2 vs 3 - Tukey adjusted ({label}) ---\n")
print(contrast(emm_block_marg, method="pairwise", adjust="tukey"))
cat("\n--- Block 1 vs 2, 1 vs 3, 2 vs 3 - Uncorrected ({label}) ---\n")
print(contrast(emm_block_marg, method="pairwise", adjust="none"))

cat("\n=== Block contrasts within each Group ({label}) ===\n")
cat("Follows up the Group x Block interaction: does each group's bias\n")
cat("change across blocks 1-2-3 on its own?\n")
emm_within_grp <- emmeans(model, ~ block_fac | Group_Enc)
cat("\n--- Block 1 vs 2, 1 vs 3, 2 vs 3 within Group - Tukey adjusted ({label}) ---\n")
print(contrast(emm_within_grp, method="pairwise", adjust="tukey"))
cat("\n--- Block 1 vs 2, 1 vs 3, 2 vs 3 within Group - Uncorrected ({label}) ---\n")
print(contrast(emm_within_grp, method="pairwise", adjust="none"))

cat("\n=== Random Effect Variance ({label}) ===\n")
print(VarCorr(model))

cat("\n=== SENSITIVITY: alternative random-effects structures ({label}) ===\n")
cat("Same fixed effects. Main = model above.\n")
cat("S1 = maximal (1 + block_fac|Participant_ID); S2 = linear block slope;\n")
cat("S3 = main model with (1|film) instead of fixed film.\n")
cat("Point estimates are identical across structures; only SEs / df / p change.\n")
sens <- list(
  main = model,
  S1_maximal  = update(model, . ~ Group_Enc * block_fac + film + (1 + block_fac|Participant_ID)
                                  + (1|Participant_ID:film)),
  S2_linear   = update(model, . ~ Group_Enc * block_fac + film + (1 + block_num|Participant_ID)
                                  + (1|Participant_ID:film)),
  S3_randfilm = update(model, . ~ Group_Enc * block_fac + (1|Participant_ID)
                                  + (1|Participant_ID:block_fac) + (1|Participant_ID:film) + (1|film))
)
for (nm in names(sens)) fit_check(sens[[nm]], nm)
sens_tab <- do.call(rbind, lapply(names(sens), function(nm) {{
  m  <- sens[[nm]]
  a  <- anova(m)
  cb <- as.data.frame(contrast(emmeans(m, ~ Group_Enc | block_fac),
                               method="pairwise", adjust="tukey"))
  cw <- as.data.frame(contrast(emmeans(m, ~ block_fac | Group_Enc),
                               method="pairwise", adjust="tukey"))
  data.frame(model = nm,
             p_Group      = a["Group_Enc", "Pr(>F)"],
             p_Block      = a["block_fac", "Pr(>F)"],
             p_GxB        = a["Group_Enc:block_fac", "Pr(>F)"],
             df_GxB       = a["Group_Enc:block_fac", "DenDF"],
             SpyCrim_B3   = cb$p.value[cb$block_fac == "3" & cb$contrast == "Spy_Pro - Crim_Pro"],
             CtrlCrim_B3  = cb$p.value[cb$block_fac == "3" & cb$contrast == "Control - Crim_Pro"],
             Crim_B1vB3   = cw$p.value[cw$Group_Enc == "Crim_Pro" & cw$contrast == "block_fac1 - block_fac3"])
}}))
num <- sapply(sens_tab, is.numeric); sens_tab[num] <- signif(sens_tab[num], 3)
cat("p-values (Tukey-adjusted within each family for the contrasts):\n")
print(sens_tab, row.names=FALSE)
"""

_R_SANITY_TEMPLATE = r"""
suppressPackageStartupMessages({{
  library(lme4); library(lmerTest); library(emmeans)
}})
options(warn=-1)
emm_options(lmer.df="satterthwaite")
""" + _R_FIT_CHECK + r"""
df <- read.csv("{csv_path}")
df$Group_Full <- factor(df$Group_Full,
                        levels=c("Control","Spy_Pro","Crim_Pro","Spy_Retro","Crim_Retro"))
df$block_fac  <- factor(df$block_fac,  levels=c("1","2","3"))
df$Participant_ID <- factor(df$Participant_ID)
df$film       <- factor(df$film)

# Same film term and random-effects structure as the Section 6 main model (_R_RE_MAIN);
# block is not a fixed effect here, but the repeated person x block and
# person x film observations still need their random terms.
model <- lmer(Bias_Score ~ Group_Full + {re},
              data=df, REML=TRUE)
cat("\n=== Singular fit / convergence check ({label}) ===\n")
fit_check(model, "sanity model")

cat("\n=== Random Effect Variance ({label}) ===\n")
print(VarCorr(model))

cat("\n=== ANOVA incl. film F-test ({label}) ===\n")
print(anova(model))

emm <- emmeans(model, ~ Group_Full)
cat("\n=== SANITY CHECK - Retro vs Control and Retro vs Retro ({label}) ===\n")
cat("Factor levels: Control, Spy_Pro, Crim_Pro, Spy_Retro, Crim_Retro\n")
cat("Unadjusted - three planned comparisons\n")
print(contrast(emm, list(
  SpyRetro_vs_Control  = c(-1, 0, 0,  1,  0),
  CrimRetro_vs_Control = c(-1, 0, 0,  0,  1),
  SpyRetro_vs_CrimRetro = c( 0, 0, 0,  1, -1)
), adjust="none"))

cat("\n=== SANITY CHECK - Descriptives ({label}) ===\n")
print(summary(emm))
"""


def _run_r_model(data: pd.DataFrame, measure_label: str,
                 dv_col: str = 'Bias_Score', sanity: bool = False,
                 group_col: str = 'Group_Enc'):
    """Print descriptives, then fit the Section 6 model (or, with sanity=True,
    the Section 7 model) in R and print its output."""
    print(f"\n--- {measure_label} ---")

    # Descriptive stats (Python side)
    desc = (data.groupby(group_col, observed=True)[dv_col]
            .agg(count='count', mean='mean', std='std',
                 min='min', max='max', median='median'))
    print(desc.round(4))

    if not HAS_R:
        print(f"  [Model skipped - Rscript not found at {RSCRIPT}]")
        return
    if data[dv_col].notna().sum() < 20:
        print("  [Model skipped - insufficient data]")
        return

    template = _R_SANITY_TEMPLATE if sanity else _R_MODEL_TEMPLATE
    result = run_r(template, data, RSCRIPT, dv=dv_col, label=measure_label, re=_R_RE_MAIN)
    print_r_output(result)


for meas in df_enc['measure'].unique():
    data_m = df_enc[df_enc['measure'] == meas].copy()
    with _log.measure_scope(meas):
        _run_r_model(data_m, meas)


# ==================================================================
# SECTION 7 - SANITY CHECK (Retro vs Control)
# ------------------------------------------------------------------
# Purpose: validate that collapsing Retro + Neutral into "Control"
# is justified. Participants in the Retro condition received their
# role instruction AFTER watching the film, so the instruction could
# not have influenced their encoding gaze. They should therefore
# look at faces and places in the same proportions as the Neutral group.
#
# Test: Spy_Retro vs Crim_Retro contrast (single test, unadjusted).
# A NON-SIGNIFICANT result is the desired outcome - it means the
# retrospective instruction had no effect on encoding gaze, which:
#   1. Validates the Control group composition
#   2. Supports the paper's claim that effects in Spy_Pro / Crim_Pro
#      are specifically due to prospective instructions
#
# If this contrast IS significant, the Control group composition
# should be reconsidered (separate Retro from Neutral).
#
# Film enters as a fixed effect (as in Section 6); its F-test in the
# ANOVA table shows how much film content shifts the bias score.
# ==================================================================
print("\n" + "="*60)
print("SECTION 7 - Sanity Check (Retro vs Control)")
print("="*60)


def _assign_group_full(timing: str, job: str) -> str:
    if timing == 'Pro'   and job == 'Spy':      return 'Spy_Pro'
    if timing == 'Pro'   and job == 'Criminal': return 'Crim_Pro'
    if timing == 'Retro' and job == 'Spy':      return 'Spy_Retro'
    if timing == 'Retro' and job == 'Criminal': return 'Crim_Retro'
    return 'Control'


df_check = df_clean[df_clean['Keep']].copy()
df_check['Group_Full'] = df_check.apply(
    lambda r: _assign_group_full(r['timing'], r['job']), axis=1)
df_check['Group_Full'] = pd.Categorical(
    df_check['Group_Full'],
    categories=['Control', 'Spy_Pro', 'Crim_Pro', 'Spy_Retro', 'Crim_Retro'],
)
df_check['Bias_Score'] = 2.0 * df_check['DV_Value'] - 1.0
df_check = df_check.rename(columns={'film_id': 'film'})
df_check['film']      = df_check['film'].astype(str)
df_check['block_fac'] = df_check['block'].astype(str)

print("\n=== SANITY CHECK: Retro Spy vs. Retro Criminal ===")
for meas in df_check['measure'].unique():
    data_m = df_check[df_check['measure'] == meas].copy()
    with _log.measure_scope(meas):
        _run_r_model(data_m, meas, dv_col='Bias_Score',
                     sanity=True, group_col='Group_Full')



# ==================================================================
# SECTION 10 - EXPORT: PARTICIPANT-LEVEL BIAS SCORES FOR RECALL CORRELATION
# ------------------------------------------------------------------
# One row per participant, to correlate
# against the free recall data. Bias scores are averaged across all
# films and blocks (kept after outlier removal in Section 2).
#
# Columns:
#   Participant_ID
#   job, timing        - condition labels (timing = instruction timing:
#                        Pro/Retro/Neutral)
#   Group_Full          - 5-level group (Control/Spy_Pro/Crim_Pro/
#                        Spy_Retro/Crim_Retro)
#   bias_<measure>      - mean Bias_Score per participant, one column
#                        per ET measure (fixation_count, fixation_duration,
#                        sample_proportion)
#   n_<measure>          - number of (film x block) observations averaged
#                        into that participant's bias score, per measure
#                        (QA: low n = fewer usable clips after cleaning)
#
# Saved to: RESULTS_DIR/eyetracking_results/ET_bias_scores_participant.csv
#           RESULTS_DIR/eyetracking_results/ET_bias_scores_participant_block.csv
# ==================================================================
print("\n" + "="*60)
print("SECTION 10 - Export Participant-Level Bias Scores")
print("="*60)

id_cols_export = ['Participant_ID', 'job', 'timing', 'Group_Full']

agg_export = (
    df_check.groupby(id_cols_export + ['measure'], observed=True)['Bias_Score']
    .agg(mean='mean', n='count')
    .reset_index()
)

df_bias_wide = (
    agg_export.pivot(index=id_cols_export, columns='measure', values='mean')
    .rename(columns=lambda m: f'bias_{m}')
    .reset_index()
)
df_n_wide = (
    agg_export.pivot(index=id_cols_export, columns='measure', values='n')
    .rename(columns=lambda m: f'n_{m}')
    .reset_index()
)
df_export = df_bias_wide.merge(df_n_wide, on=id_cols_export)
df_export.columns.name = None

out_path = os.path.join(PATH_OUT, 'ET_bias_scores_participant.csv')
df_export.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")
print(f"  {len(df_export)} participants x {df_export.shape[1]} columns")
print(df_export.head())

# --- Block-level export (long): one row per participant x block ----
# Same aggregation but keeps `block` in the key instead of collapsing
# over it, so block-wise effects (e.g. the block-3 group separation)
# can be inspected / correlated. Bias scores still averaged across
# films within each block.
agg_block = (
    df_check.groupby(id_cols_export + ['block', 'measure'], observed=True)['Bias_Score']
    .agg(mean='mean', n='count')
    .reset_index()
)
idx_block = id_cols_export + ['block']
df_bias_block = (
    agg_block.pivot(index=idx_block, columns='measure', values='mean')
    .rename(columns=lambda m: f'bias_{m}')
    .reset_index()
)
df_n_block = (
    agg_block.pivot(index=idx_block, columns='measure', values='n')
    .rename(columns=lambda m: f'n_{m}')
    .reset_index()
)
df_export_block = df_bias_block.merge(df_n_block, on=idx_block)
df_export_block.columns.name = None

out_path_block = os.path.join(PATH_OUT, 'ET_bias_scores_participant_block.csv')
df_export_block.to_csv(out_path_block, index=False)
print(f"  Saved: {out_path_block}")
print(f"  {len(df_export_block)} participant x block rows x {df_export_block.shape[1]} columns")
print(df_export_block.head())



# ==================================================================
# SECTION 11 - CATEGORY VIEWING TIME BREAKDOWN (completeness report)
# ------------------------------------------------------------------
# Sections 4-10 all report face_prop = face / (face + place): objects
# and uncategorised ("others") fixations are excluded from that ratio's
# denominator by design (see Section 1 header), so the bias score never
# says anything about how much time was spent on faces/places in
# absolute terms, or how much went to the other two semantic categories.
#
# This section reports, for completeness, each of the four semantic
# categories (personal/face, spatial/place, object, others) as a share
# of ALL valid viewing (all categories combined) - i.e. what fraction
# of total fixations/dwell time/gaze samples actually landed on each
# category. A fifth bucket, 'unassigned', catches valid (in-bounds)
# fixations/samples whose category didn't match any of the four labels
# (not a semantic category: for samples these are blinks and brief
# tracking loss, with no gaze position to look up). It's ~0% for fixations
# and ~2-4% for raw gaze samples (see _aggregate_sample). All five shares sum to
# ~100% per observation.
#
# Same reliability gate as Section 2 (>= MIN_VALID_FIX fixations for
# fixation_count/fixation_duration, >= MIN_VALID_SAMPLE samples for
# sample_proportion), applied to the ALL-category valid total rather
# than the face+place-only total - a strictly looser bar, since the
# all-category total is always >= the face+place-only total.
#
# Purely descriptive (no mixed models) - this is a reporting table for
# the paper's methods/results, not a hypothesis test.
#
# Saved to: RESULTS_DIR/eyetracking_results/ET_category_breakdown_participant.csv
#           RESULTS_DIR/eyetracking_results/ET_category_breakdown_summary.csv
#           RESULTS_DIR/ET_category_breakdown_radar.png (paper figure)
# ==================================================================
print("\n" + "="*60)
print("SECTION 11 - Category Viewing Time Breakdown")
print("="*60)

CATEGORY_PALETTE = {
    'face':       '#0072B2',   # matches Spy_Pro in GROUP_PALETTE (hypothesised face preference)
    'place':      '#D55E00',   # matches Crim_Pro in GROUP_PALETTE (hypothesised place preference)
    'object':     '#009E73',
    'others':     '#999999',
    'unassigned': '#DDDDDD',   # data-quality artifact, not a semantic category -
                               # ~0% for fixations, ~1-2% for samples (see _aggregate_sample)
}
_CAT_LIST = ['face', 'place', 'object', 'others', 'unassigned']

# measure -> (share-column suffix, all-category valid-total column)
_SHARE_COL_BY_MEASURE = {
    'fixation_count':    ('count_share',  'valid_fix_count'),
    'fixation_duration':  ('dur_share',    'valid_fix_dur_ms'),
    'sample_proportion': ('sample_share', 'valid_sample_count'),
}

_cat_rows = []
for meas, (suffix, valid_col) in _SHARE_COL_BY_MEASURE.items():
    for cat_name in _CAT_LIST:
        col = f'{cat_name}_{suffix}'
        sub = df_wide[id_cols + [col, valid_col]].copy()
        sub = sub.rename(columns={col: 'Share', valid_col: 'N_Valid_Total'})
        sub['measure']  = meas
        sub['category'] = cat_name
        _cat_rows.append(sub)

df_cat_long = pd.concat(_cat_rows, ignore_index=True)
df_cat_long = df_cat_long.dropna(subset=['Share', 'job', 'timing'])
print(f"  Long-format dataset: {len(df_cat_long)} rows")

df_cat_long['Min_Required'] = df_cat_long['measure'].map(_MIN_VALID_BY_MEASURE)
df_cat_long['Keep']         = df_cat_long['N_Valid_Total'] >= df_cat_long['Min_Required']
n_removed_cat = (~df_cat_long['Keep']).sum()
print(f"  Observations removed (reliability gate): {n_removed_cat} "
      f"({n_removed_cat/len(df_cat_long)*100:.2f}%)")

df_cat_enc = df_cat_long[df_cat_long['Keep']].copy()
df_cat_enc['Group_Enc'] = df_cat_enc.apply(
    lambda r: _assign_group_enc(r['timing'], r['job']), axis=1)
df_cat_enc['Group_Enc'] = pd.Categorical(
    df_cat_enc['Group_Enc'], categories=['Control', 'Spy_Pro', 'Crim_Pro'])

# --- Descriptive table: mean share (%) per measure x group x category ---
# Collapsed across films and blocks, mirroring the Section 10 export.
desc_cat = (
    df_cat_enc.groupby(['measure', 'Group_Enc', 'category'], observed=True)['Share']
    .agg(Mean='mean', SD='std', N='count')
    .reset_index()
)
desc_cat['Mean_%'] = (desc_cat['Mean'] * 100).round(2)
desc_cat['SD_%']    = (desc_cat['SD'] * 100).round(2)

print("\n  --- Category viewing time share (%) by measure x group ---")
for meas in _SHARE_COL_BY_MEASURE:
    print(f"\n  {meas}:")
    tbl = (desc_cat[desc_cat['measure'] == meas]
           .pivot(index='Group_Enc', columns='category', values='Mean_%')
           .reindex(columns=_CAT_LIST))
    print(tbl.to_string())


# --- Participant-level export (mirrors Section 10's export pattern) ---
id_cols_cat_export = ['Participant_ID', 'job', 'timing', 'Group_Enc']
agg_cat_export = (
    df_cat_enc.groupby(id_cols_cat_export + ['measure', 'category'], observed=True)['Share']
    .agg(mean='mean', n='count')
    .reset_index()
)
agg_cat_export['col'] = agg_cat_export['measure'] + '_' + agg_cat_export['category']
df_cat_wide = (
    agg_cat_export.pivot(index=id_cols_cat_export, columns='col', values='mean')
    .rename(columns=lambda c: f'share_{c}')
    .reset_index()
)
df_cat_wide.columns.name = None

out_path = os.path.join(PATH_OUT, 'ET_category_breakdown_participant.csv')
df_cat_wide.to_csv(out_path, index=False)
print(f"  Saved: {out_path}")
print(f"  {len(df_cat_wide)} participants x {df_cat_wide.shape[1]} columns")
print(df_cat_wide.head())

# --- Plot: radial (radar) category profile, averaged across blocks & films --
# One radar panel per ET measure. Axes = the five categories (closed
# pentagon); one line per group = mean share (%) with a shaded mean +/- SEM
# ribbon. Built from agg_cat_export (already averaged across blocks/films
# per participant, same as the CSV export above), then averaged again across
# participants within each group (mirrors the radar plots used for the
# free-recall categories).
radar_summary = (
    agg_cat_export.rename(columns={'mean': 'ParticipantShare'})
    .groupby(['measure', 'Group_Enc', 'category'], observed=True)['ParticipantShare']
    .agg(Mean='mean', SD='std', N='count')
    .reset_index()
)
radar_summary['SEM']   = radar_summary['SD'] / np.sqrt(radar_summary['N'])
radar_summary['Mean_%'] = radar_summary['Mean'] * 100
radar_summary['SEM_%']  = radar_summary['SEM'] * 100

# --- Manuscript-ready summary table: N, Mean, SD, SEM, 95% CI (%) ---
# Same participant-level basis as radar_summary (one value per participant,
# already averaged across blocks/films), so N/SEM aren't pseudo-replicated
# across the block x film rows the way the earlier desc_cat table's N is.
# 95% CI uses the t-distribution (df = N-1) rather than a fixed 1.96,
# since group N here is modest. Purely descriptive - no group comparison.

cat_summary = radar_summary.copy()
cat_summary['SD_%'] = cat_summary['SD'] * 100
_tcrit = _sp_stats.t.ppf(0.975, (cat_summary['N'] - 1).clip(lower=1))
_halfwidth = np.where(cat_summary['N'] > 1, _tcrit * cat_summary['SEM_%'], np.nan)
cat_summary['CI95_low_%']  = (cat_summary['Mean_%'] - _halfwidth).clip(lower=0)
cat_summary['CI95_high_%'] = (cat_summary['Mean_%'] + _halfwidth).clip(upper=100)
cat_summary = cat_summary[['measure', 'Group_Enc', 'category', 'N',
                            'Mean_%', 'SD_%', 'SEM_%', 'CI95_low_%', 'CI95_high_%']]
for col in ['Mean_%', 'SD_%', 'SEM_%', 'CI95_low_%', 'CI95_high_%']:
    cat_summary[col] = cat_summary[col].round(2)

print("\n  --- Manuscript summary: category share (%) by measure x group "
      "(participant-level N, mean, SD, SEM, 95% CI) ---")
for meas in _SHARE_COL_BY_MEASURE:
    print(f"\n  {meas}:")
    print(cat_summary[cat_summary['measure'] == meas]
          .drop(columns='measure').to_string(index=False))

out_path = os.path.join(PATH_OUT, 'ET_category_breakdown_summary.csv')
cat_summary.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

# --- Statistical test: does viewing time differ across categories? ---
# Supports the manuscript claim that participants allocate most viewing
# time to faces, then places, then objects. Friedman test (non-parametric
# repeated-measures ANOVA analogue) rather than RM-ANOVA, since category
# shares are compositional/bounded and not guaranteed normal. Collapsed
# across group and block (agg_cat_export is already one row per
# participant, averaged across blocks/films) -- this establishes the
# overall gaze pattern, not a group difference (that's Section 6's
# Group x block mixed model on the face/place bias score).
# 'unassigned' is excluded: a data-quality artifact, not a semantic
# category (see Section 11 header).
_FRIEDMAN_CATS = ['face', 'place', 'object', 'others']

print("\n  --- Friedman test: do category shares differ overall? "
      "(participant-level, collapsed across group & block) ---")
for meas in _SHARE_COL_BY_MEASURE:
    wide = (agg_cat_export[(agg_cat_export['measure'] == meas)
                           & (agg_cat_export['category'].isin(_FRIEDMAN_CATS))]
            .pivot(index='Participant_ID', columns='category', values='mean')
            .reindex(columns=_FRIEDMAN_CATS)
            .dropna())
    stat, p = _sp_stats.friedmanchisquare(*[wide[c] for c in _FRIEDMAN_CATS])
    df_fr = len(_FRIEDMAN_CATS) - 1
    print(f"  {meas}: n={len(wide)}, chi2({df_fr})={stat:.2f}, p={p:.4g}")

n_cat_ax   = len(_CAT_LIST)
angles     = np.linspace(0, 2 * np.pi, n_cat_ax, endpoint=False).tolist()
angles    += angles[:1]
cat_labels = [c.capitalize() for c in _CAT_LIST]

# "Flower" shape, after the ggplot coord_polar plots, with one
# petal per category: the outline rises from the centre to the category's
# value on its axis and falls back to the centre before the next category,
# r(theta) = value * (1 - u**PETAL_TIP), u = d / half_width, d = angular
# distance from the category axis. PETAL_TIP <= 1 gives a pointed tip
# (1 = straight-sided point, lower = sharper, more needle-like); values > 1
# round it off. PETAL_WIDTH = fraction of each category's sector the petal
# spans (smaller = thinner petals). Petal tips are the data values.
PETAL_WIDTH = 0.85
PETAL_TIP   = 1.0



# Small per-group angular dodge so the three groups' overlapping outlines
# separate visually around each category axis instead of sitting exactly on
# top of each other, while the axis tick labels stay at the true angle.
_dodge_step = (2 * np.pi / n_cat_ax) * 0.12
_group_dodge = {
    grp: (gi - (len(group_order) - 1) / 2) * _dodge_step
    for gi, grp in enumerate(group_order)
}

_MEASURE_TITLES = {'fixation_count': 'Fixation count',
                   'fixation_duration': 'Fixation duration',
                   'sample_proportion': 'Sample proportion'}

_radar_rc = plt.rc_context(PAPER_RC)
_radar_rc.__enter__()
fig, axes_radar = plt.subplots(1, len(_SHARE_COL_BY_MEASURE),
                                figsize=(FIG_WIDTH_IN, 2.5),
                                subplot_kw=dict(polar=True))

for j, meas in enumerate(_SHARE_COL_BY_MEASURE):
    ax = axes_radar[j]
    meas_data = radar_summary[radar_summary['measure'] == meas]
    for grp in group_order:
        tbl = (meas_data[meas_data['Group_Enc'] == grp]
               .set_index('category').reindex(_CAT_LIST))
        mean_vals    = tbl['Mean_%'].to_numpy()
        sem_vals     = tbl['SEM_%'].to_numpy()
        mean_closed  = np.append(mean_vals, mean_vals[0])
        upper_closed = mean_closed + np.append(sem_vals, sem_vals[0])
        lower_closed = np.clip(mean_closed - np.append(sem_vals, sem_vals[0]), 0, None)
        color = GROUP_PALETTE[grp]
        dodge = _group_dodge[grp]
        theta_d, mean_d = polar_interp(mean_closed, n_cat_ax, PETAL_WIDTH, PETAL_TIP, dodge)
        _, lower_d = polar_interp(lower_closed, n_cat_ax, PETAL_WIDTH, PETAL_TIP, dodge)
        _, upper_d = polar_interp(upper_closed, n_cat_ax, PETAL_WIDTH, PETAL_TIP, dodge)

        ax.plot(theta_d, mean_d, color=color, linewidth=1.0, label=GROUP_LABELS[grp])
        ax.plot(np.asarray(angles) + dodge, mean_closed, linestyle='none', marker='o',
                markersize=2, color=color)
        ax.fill_between(theta_d, lower_d, upper_d, color=color, alpha=0.25, linewidth=0)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=6)
    meas_max = (meas_data['Mean_%'] + meas_data['SEM_%']).max()
    ax.set_ylim(0, meas_max * 1.15 if pd.notna(meas_max) and meas_max > 0 else 1)
    ax.set_yticks([20, 40, 60])
    ax.set_rlabel_position(180)      # radius labels along the empty gap between Object and Others
    ax.tick_params(axis='y', labelsize=5, colors='grey')
    ax.tick_params(axis='x', pad=-2)
    ax.grid(color='#e5e5e5', linewidth=0.6)
    ax.spines['polar'].set_visible(False)
    ax.set_title(_MEASURE_TITLES.get(meas, meas), pad=12)

handles_radar = [Patch(facecolor=GROUP_PALETTE[g], label=GROUP_LABELS[g]) for g in group_order]
fig.legend(handles=handles_radar, loc='upper center', ncol=len(group_order),
           frameon=False, bbox_to_anchor=(0.5, 0.12))
# fig.suptitle(#'Encoding: Category Viewing-Time Profile by Group\n'
#              r'Solid line = mean %  |  shaded band = mean $\pm$ SEM  |  averaged across blocks & films',
#              fontsize=10)
fig.tight_layout()
out_path = _fig_path('ET_category_breakdown_radar.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
_radar_rc.__exit__(None, None, None)
print(f"\n  Saved: {out_path}")


print("\n" + "="*60)
print("DONE")
print(f"Results saved to: {PATH_OUT}")
print("="*60)
_log.close()
