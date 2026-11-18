"""Copy this file to config.py and edit the paths."""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Root of the unpacked data deposit
DATA_ROOT = r"path/to/data_deposit"
ET_DERIVATIVES = os.path.join(DATA_ROOT, "derivatives", "eyetracking")
BEHAV_ENCODING_CSV = os.path.join(DATA_ROOT, "behavioural_results", "shamem_encoding_behav.csv")
CONDITIONS_CSV = os.path.join(DATA_ROOT, "study_info", "conditions.csv")

# Where figures, logs and tables are written
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# Rscript: a command on your PATH or a full path to the executable
RSCRIPT = "Rscript"

# Only for the pipeline and segmentation. These inputs are not in the deposit.
VIDEO_DIR = r"path/to/films"                    # the six .mp4 clips
FRAMES_DIR = r"path/to/frames"                  # written by eyetracking/segmentation/extract_frames.py
MASKS_DIR = r"path/to/masks"                    # written by eyetracking/segmentation/segment_frames.py
SEGMENTATION_WEIGHTS_DIR = r"path/to/weights"   # encoder_epoch_50.pth, decoder_epoch_50.pth
RAW_ASC_DIR = r"path/to/asc"                    # raw EyeLink .asc recordings
PIPELINE_WORK_DIR = r"path/to/pipeline_work"    # intermediate pipeline files
CATEGORY_MAPPING_FILE = os.path.join(REPO_ROOT, "eyetracking", "pipeline", "category_mapping.xlsx")
PARTICIPANT_SHEET = None                        # optional and lab-internal; leave as None
