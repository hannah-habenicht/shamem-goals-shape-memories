# How goals shape naturalistic memories: analysis code

Code for the paper **How goals shape naturalistic memories: bias in gaze and narrative recall** by Habenicht, Müller, Schönauer and Roüast.

- **Preregistration:** AsPredicted #261319
- **Data:** [10.5281/zenodo.24275394](https://doi.org/10.5281/zenodo.24275394)
- **Licence:** code under MIT ([LICENSE](LICENSE)). The segmentation model and its code are third-party, under their own licence (BSD 3-Clause).

## The study in brief

100 participants watched six film clips in each of three blocks while gaze was recorded at 500 Hz (EyeLink 1000 Plus). Each participant was given a goal state, spy (goal: people) or criminal (goal: places), either before encoding (proactive) or after it (retroactive), or had no role (neutral). During encoding they detected coins appearing on faces, places or objects, and in a second session they freely recalled the films.

## Repository layout

| Folder | Contents | README |
|---|---|---|
| [`eyetracking/`](eyetracking/) | Film segmentation, gaze-to-label pipeline, encoding eye-tracking analyses | [eyetracking/README.md](eyetracking/README.md) |
| [`behavioural/`](behavioural/) | Coin detection, free recall, correlations between attention and narrative bias | [behavioural/README.md](behavioural/README.md) |
| [`study_info/`](study_info/) | Description of the data deposit | [study_info/DATA_README.md](study_info/DATA_README.md) |

## What you can reproduce

Films are commercial material and raw eyetracking recordings are not public, so the starting point depends on what you want to do.

| Goal | You need | Public? |
|---|---|---|
| All statistics and figures in the paper (except those showing film frames) | Data deposit, Python, R | Yes |
| Segmentation masks | Film frames, model weights, GPU recommended | Frames no |
| Pipeline from raw gaze | Raw `.asc` files, masks, films | No |

The clips were cut from Charité, Ziemlich beste Freunde, High Seas, Biohackers, Downton Abbey and New Amsterdam; sources and timestamps are listed in `study_info/stimuli.csv` in the data deposit. The clips themselves are copyrighted and cannot be shared publicly.

## Installation

Requires Python 3.11 and R 4.4 or later.

```bash
git clone https://github.com/hannah-habenicht/shamem-goals-shape-memories.git
cd shamem-goals-shape-memories
conda env create -f environment.yml
conda activate shamem1
cp config.example.py config.py    # then edit the paths in config.py
```

In R, install the packages listed in `R_requirements.txt`:

```r
install.packages(c("lme4", "lmerTest", "emmeans"))
# for behavioural/ also:
install.packages(c("car", "dplyr", "tidyr", "ggplot2", "scales", "MASS",
                   "sandwich", "lmtest", "betareg", "ggdist", "patchwork"))
```

## Reproducing the results

1. Download and unpack the data deposit, and set `DATA_ROOT` and `RSCRIPT` in `config.py`.
2. Run the eye-tracking analyses ([eyetracking/README.md](eyetracking/README.md)). They also write the bias scores the correlations need.
3. Run the behavioural analyses ([behavioural/README.md](behavioural/README.md)).

## Data

The data are on Zenodo: [10.5281/zenodo.24275394](https://doi.org/10.5281/zenodo.24275394). File formats and column descriptions for every file are in the deposit's README, also in this repository as [`study_info/DATA_README.md`](study_info/DATA_README.md).

## Contact

Please contact the corresponding author named in the paper.

## Citation

If you use this code, please cite the paper and the archived release (see [CITATION.cff](CITATION.cff)).
