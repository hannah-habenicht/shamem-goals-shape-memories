# Utilities and configuration for the pipeline
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config

# Constants
BLOCKS = 3
ORDERS = 6


def setup_logging(level=logging.INFO):
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger('eyetracking_pipeline')


# Paths are set in config.py at the repository root
INPUT_DIR = config.RAW_ASC_DIR
OUTPUT_DIR_04 = config.PIPELINE_WORK_DIR
DERIVATIVES_ROOT = config.ET_DERIVATIVES
VIDEO_DIR = config.VIDEO_DIR
SEGMENTATION_DIR = config.MASKS_DIR
CATEGORY_MAPPING_FILE = config.CATEGORY_MAPPING_FILE

# Movie files mapping (key -> filename). Update if your filenames differ.
MOVIE_FILES = {
    "movie_01": "Charite.mp4",
    "movie_02": "Ziemlich_Beste_Freunde.mp4",
    "movie_03": "High_Seas.mp4",
    "movie_04": "Biohackers.mp4",
    "movie_05": "Downton_Abbey.mp4",
    "movie_06": "New_Amsterdam.mp4",
}
