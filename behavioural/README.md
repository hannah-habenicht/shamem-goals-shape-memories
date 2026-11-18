# Behavioural, free-recall and correlation analyses

The scripts here cover coin detection during encoding (accuracy and reaction time), free recall, and the correlations between attention bias and narrative bias.

## Scripts

| Script | Language | Role | Input | Output |
|---|---|---|---|---|
| `beh_ana01_encoding_readin.m` | MATLAB | Reference only | Raw encoding `.xlsx` files (not public) | `behavioural_results/shamem_encoding_behav.csv`, `shamem_exclusion_behav.csv` |
| `beh_ana02_memory_readin.m` | MATLAB | Reference only | Raw retrieval `.xlsx` files (not public) | `behavioural_results/shamem_retrieval_behav.csv` (spatial placement; not reported in the paper) |
| `beh_ana03_freerecall_readin.m` | MATLAB | Optional | MAXQDA statistics per participant (`rawdata/freerecall/MAXQDA_output/` in the data deposit), `conditions.csv` | `behavioural_results/shamem_retrieval_freerecall.csv` |
| `beh_ana04_encoding_analysis.R` | R | Analysis | `behavioural_results/shamem_encoding_behav.csv`, `conditions.csv` | Detection statistics (console) and figures |
| `beh_ana05_freerecall_analysis.R` | R | Analysis | `behavioural_results/shamem_retrieval_freerecall.csv`, eye-tracking bias scores | Free-recall and correlation statistics (console and `.csv`) and figures |

The MATLAB scripts document how the deposited `.csv` files were made. `beh_ana01` and `beh_ana02` read raw files that are not public. `beh_ana03` reads the MAXQDA statistics, which are in the data deposit, so the free-recall file can be rebuilt from them; it overwrites `behavioural_results/shamem_retrieval_freerecall.csv`. None of the MATLAB scripts are needed to reproduce the results: start from the data deposit and run the two R scripts.

## Running

1. Install the R packages in [`../R_requirements.txt`](../R_requirements.txt).
2. In each R script, set the two lines at the top:
   - `DATA_ROOT`: the unpacked data deposit.
   - `RESULTS_DIR`: where outputs go, for example the repository's `results/` folder. Outputs are written to `RESULTS_DIR/behavioural/`.
3. For the correlations, `beh_ana05` reads `ET_bias_scores_participant.csv` and `ET_bias_scores_participant_block.csv` from `RESULTS_DIR/eyetracking_results/`. Run `eyetracking/analysis/encoding_bias.py` first to create them, or copy the versions used in the paper from the data deposit.
4. Run `beh_ana04_encoding_analysis.R` and `beh_ana05_freerecall_analysis.R` in RStudio. Both print their statistics to the console. Run from the command line with `Rscript`, `beh_ana05` stops at the first radial plot with `invalid font type`, because the plots use Arial and the default command-line graphics device does not have it.

Group assignment comes from `conditions.csv`: `beh_ana04` reads it directly, and `beh_ana05` uses the condition codes in `shamem_retrieval_freerecall.csv`, which `beh_ana03` copied from it. Participant exclusions are applied inside the scripts and printed to the console.

## Where each result comes from

| Paper section | Script | Part |
|---|---|---|
| Detection task: accuracy (binomial GLMM) and reaction time (LMM) | `beh_ana04` | Whole script |
| Free recall: counts per category (negative binomial GLM) | `beh_ana05` | Part 1 |
| Free recall: percentages per category (beta regression) | `beh_ana05` | Part 2 |
| Narrative bias score | `beh_ana05` | Part 3 (Part 3b: normalised score, confirmatory) |
| Correlations between attention bias and narrative bias (Spearman) | `beh_ana05` | Part 4 |

## Figure

`figure_freerecall.pdf` in the paper was assembled by hand from SVGs written by `beh_ana05`:

| Panel | File |
|---|---|
| A, B | `radial_plot_pro_*.svg`, `radial_plot_retro_*.svg` |
| C | `shamem1_bias_scores_*_nooutlier.svg` |
| D | `plot_bias_*_proactive_only.svg` |
| E, F | `plot_correlation_bias_vs_slope_count_pro.svg`, `plot_correlation_bias_vs_slope_proportion_pro.svg` |
