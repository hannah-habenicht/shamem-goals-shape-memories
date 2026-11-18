
## Contents


```
derivatives/eyetracking/sub-XXX/ses-01/eyetrack/           BIDS gaze files, 95 participants
rawdata/behavioural/                                       Raw PsychoPy output: three encoding blocks and one retrieval file per participant
rawdata/freerecall/MAXQDA_output/                          MAXQDA code statistics, one file per participant
behavioural_results/shamem_encoding_behav.csv              Trial-level detection data
behavioural_results/shamem_retrieval_freerecall.csv        Free-recall counts and percentages per category
behavioural_results/shamem_retrieval_behav.csv             Spatial placement at retrieval (not reported in the paper)
behavioural_results/shamem_exclusion_behav.csv             Coins never detected during encoding, excluded from placement
eyetracking_results/ET_bias_scores_participant.csv         Eye-tracking bias scores used for the correlations
eyetracking_results/ET_bias_scores_participant_block.csv   (copies of encoding_bias.py output)
study_info/conditions.csv                                  Group assignment
study_info/exclusions.csv                                  Excluded participants (eye-tracking)
study_info/stimuli.csv                                     Film material specifics
segmentation_weights.zip                                   ADE20K ResNet101-UPerNet weights (third-party, BSD 3-Clause)
```

Participant numbers are the same in all files: `id` 1 is `sub-001`. Films are numbered the same way throughout: `movie_01` Charité, `movie_02` Ziemlich beste Freunde, `movie_03` High Seas, `movie_04` Biohackers, `movie_05` Downton Abbey, `movie_06` New Amsterdam. The films themselves are not included.

## Eye-tracking files

The gaze files are the output of the processing pipeline in the code repository: EyeLink samples, fixations and saccades, split into one run per film viewing, matched to video frames and labelled with the semantic category at the gaze position. Each run has four data files, each with a `.json` sidecar. `run = (block - 1) * 6 + order`.

```
sub-XXX_ses-01_task-experiment01_run-NN_recording-eye1_physio.tsv.gz
sub-XXX_ses-01_task-experiment01_run-NN_recording-eye2_physio.tsv.gz
sub-XXX_ses-01_task-experiment01_run-NN_recording-fixations_physioevents.tsv.gz
sub-XXX_ses-01_task-experiment01_run-NN_recording-saccades_physioevents.tsv.gz
```

Four things catch people out:
- **The TSV files have no header.** Column names are in the sidecar's `Columns` field.
- **`duration` is in seconds**, not milliseconds.
- **Use `eye1` only for category counts.** `eye1` and `eye2` carry identical category columns, so reading both double-counts.
- **Gaze coordinates are only in the recorded eye's file.** Only one eye was recorded per participant, so in each run either `eye1` (left) or `eye2` (right) has the `x_coordinate`/`y_coordinate` values and the other has them empty. The `eye` column in the fixation file tells you which eye was recorded.

```python
import json, pandas as pd
base = "sub-001_ses-01_task-experiment01_run-01_recording-fixations_physioevents"
cols = json.load(open(base + ".json"))["Columns"]
df = pd.read_csv(base + ".tsv.gz", sep="\t", header=None, names=cols, compression="gzip")
```

Main fixation columns:

| Column | Meaning |
|---|---|
| `onset`, `duration` | Event timing; `duration` in seconds |
| `tStart`, `tEnd` | EyeLink timestamps |
| `xAvg`, `yAvg`, `pupilAvg` | Mean gaze position (screen pixels, 1920 x 1080) and pupil size |
| `Movie` | Film as `movie_0X` |
| `block`, `order`, `run_index` | Position in the session |
| `matched_frame_IDx_start`, `matched_frame_IDx_end` | Matched video frames |
| `MatchingPixel_Start`, `MatchingPixel_End` | ADE20K class at the gaze position |
| `AssignedCategory_*`, `Category_*`, `Designation_*` | ADE20K label and collapsed category |
| `OutOfBounds` | Gaze fell outside the screen |

Sample files use `timestamp`, `x_coordinate`, `y_coordinate`, `pupil_size`, `matching_pixel`, `out_of_bounds`, `assigned_category`, `category` and `designation`. The film comes from `source_chunk_key`, for example `Block_1_Order_1_movie_03`.

## study_info/conditions.csv
One row per participant, giving the condition they were assigned to.

| Column | Values |
|---|---|
| `id` | Participant number |
| `type` | Instruction timing: 1 proactive, 2 retroactive, 3 neutral |
| `job` | Role: 1 spy, 2 criminal, 3 neutral |
| `version` | Instruction speaker: 1 female speaker 1, 2 female speaker 2, 3 no speaker (neutral) |

`type` and `job` together define five conditions:

| Condition | `type` | `job` | Role instruction | n |
|---|---|---|---|---|
| Proactive spy | 1 | 1 | Spy (goal: people), given before encoding | 20 |
| Proactive criminal | 1 | 2 | Criminal (goal: places), given before encoding | 20 |
| Retroactive spy | 2 | 1 | Spy, given after encoding, before recall | 20 |
| Retroactive criminal | 2 | 2 | Criminal, given after encoding, before recall | 21 |
| Neutral | 3 | 3 | No role | 19 |

The two speakers (`version` 1 and 2) are balanced within the proactive and retroactive conditions.

Because retroactive participants learned their role only after encoding, the encoding analyses (eye tracking and coin detection) compare three groups: proactive spy (`Spy_Pro`), proactive criminal (`Crim_Pro`) and `Control`, which combines retroactive spy, retroactive criminal and neutral. A separate check compares the two retroactive groups with the rest of `Control`. The free-recall analyses keep all five conditions separate; there, `Control` means the neutral condition only.

## behavioural_results/shamem_encoding_behav.csv

Coin detection during encoding, in long format: one row per participant, trial and measure. Every participant has 243 accuracy rows (81 coins per block, 3 blocks); RT rows exist only for detected coins. Exclusions and RT cleaning are applied in the analysis code (`behavioural/beh_ana04_encoding_analysis.R`), not in the file.

| Column | Values |
|---|---|
| `id` | Participant number |
| `session` | Always 1 (encoding) |
| `timing`, `job`, `voice` | Condition codes as in `conditions.csv`, but taken from the raw file names; they differ from `conditions.csv` for participant 14. Use `conditions.csv` |
| `block` | Encoding block, 1-3 |
| `filmID` | Film, 1-6 (`movie_01`-`movie_06`) |
| `coinID` | Coin identifier |
| `category` | Where the coin appeared: 1 face, 2 place, 3 object |
| `trialID` | Trial number within the block |
| `measure` | `acc` or `RT` |
| `output` | For `acc`: 1 detected, 0 not detected. For `RT`: reaction time in seconds, uncleaned |

## behavioural_results/shamem_retrieval_freerecall.csv

Free recall, coded in MAXQDA, in long format: one row per participant, category, correctness and measure. The coding scheme (what counts as a person, spatial, object, speech or other detail, and when a detail is wrong) is described in the paper.

| Column | Values |
|---|---|
| `id` | Participant number |
| `session` | Always 2 (retrieval) |
| `timing`, `job`, `voice` | Condition codes, taken from `conditions.csv` |
| `category` | `overall`, `unrelated` (not film-related), `insecurity` (signs of insecurity), `person`, `spatial`, `object`, `speech`, `other` |
| `correct` | `right`, `wrong`, or `total` (right and wrong combined, or a category without a right/wrong split) |
| `measure` | `count` (number of coded segments) or `percent` (share of valid coded segments, from MAXQDA) |
| `output` | Value of the measure |

## rawdata/behavioural/

The raw task output, one `.csv` per encoding block and one per retrieval session:

```
NNN_01_TT_JJ_VV_shamem_study1_encoding_B.csv     encoding, block B (1-3)
NNN_02_TT_JJ_VV_shamem_study1_retrieval.csv      retrieval (coin placement)
```

`NNN` is the participant number, the second field the session, and `TT`, `JJ`, `VV` the timing, job and voice codes from the file names. As in `shamem_encoding_behav.csv`, these differ from `study_info/conditions.csv` for participant 14; use `conditions.csv`. Participant 14 has no file for encoding block 3. Each row is one coin. Positions are in screen pixels with the origin at the screen centre; times are in seconds.

Encoding files:

| Column | Values |
|---|---|
| `film_ID` | Film, 1-6 (`movie_01`-`movie_06`) |
| `coin_number` | Coin number within the film |
| `coin_ID` | Coin identifier |
| `image` | Coin image file |
| `cond` | Where the coin appeared: 1 face, 2 place, 3 object |
| `onset_time` | Planned coin onset within the film |
| `x`, `y` | Coin position |
| `in_film_time` | Actual coin onset within the film |
| `stim_presentation_time`, `click_time` | Coin onset and response on the experiment clock |
| `RT` | `click_time - stim_presentation_time`; empty if the coin was not detected |

Retrieval files have the first eight columns of the encoding files, then:

| Column | Values |
|---|---|
| `initial_x`, `initial_y` | Start position of the coin to be placed |
| `response_x`, `response_y` | Position where it was placed; `0, 0` means no response |
| `L1_path`, `L2_path` | City-block and Euclidean distance from start to placed position |
| `L1_precision`, `L2_precision` | City-block and Euclidean distance from placed to true position (`x`, `y`) |
| `scene_time`, `coin_time`, `placement_time` | Scene onset, coin onset and placement on the experiment clock |
| `RT_coin` | `coin_time - scene_time` |
| `RT_deliberation` | `placement_time - coin_time` |

`behavioural/beh_ana01_encoding_readin.m` and `beh_ana02_memory_readin.m` in the code repository turn these files into `shamem_encoding_behav.csv`, `shamem_exclusion_behav.csv` and `shamem_retrieval_behav.csv`.

## rawdata/freerecall/MAXQDA_output/

The source of `shamem_retrieval_freerecall.csv`: one MAXQDA export per participant, `NNN_Statistik für Dokumentvariablen.xlsx`, with the frequency and percentage of each code in that participant's recall. They contain code statistics only, no transcribed text. `behavioural/beh_ana03_freerecall_readin.m` in the code repository turns them into the long-format file.

## eyetracking_results/ET_bias_scores_participant*.csv

Bias scores, `(face - place) / (face + place)`, per participant and per participant and block, as written by `eyetracking/analysis/encoding_bias.py` in the code repository. They are included so the correlations can be run without rerunning the eye-tracking analysis.

## segmentation_weights.zip

The pretrained scene-parsing weights used to label the film frames (`encoder_epoch_50.pth`, `decoder_epoch_50.pth`), from CSAIL's [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch). They are included, unchanged and under their original BSD 3-Clause licence, because the original download site was unreachable in November 2026. The zip contains the licence, MD5 checksums and the papers to cite. They are only needed to rerun the segmentation step.

## Exclusions

`study_info/exclusions.csv` lists the participants excluded from the eye-tracking analyses and the reason for each. Behavioural exclusions are applied in the analysis code and described in the paper.
