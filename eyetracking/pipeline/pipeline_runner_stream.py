"""Stream-based pipeline runner (write-once, memory-efficient).

This runner executes the full eyetracking analysis pipeline using streaming
(in-memory) processing. All intermediate data flows through memory, and only
the final semantic category assignments are written to disk.

This approach significantly reduces:
- Disk storage footprint
- Disk I/O overhead
- Total pipeline execution time
"""
from pipeline_utils import (setup_logging, INPUT_DIR, OUTPUT_DIR_04, VIDEO_DIR, 
                            SEGMENTATION_DIR, CATEGORY_MAPPING_FILE, MOVIE_FILES,
                            BLOCKS, ORDERS, DERIVATIVES_ROOT)
from pipeline_scaffold_stream import (StreamAnalysis00, StreamAnalysis01, 
                                      StreamAnalysis02, StreamAnalysis03, 
                                      StreamAnalysis04)
import config
from video_cache import VideoMetadataCache
from processing_report import (load_participant_remarks, write_processing_report,
                               session_of)
import argparse
import logging
import os
import pickle
import re
from datetime import datetime


def get_processed_participants(output_dir):
    """Get set of participants that have been fully processed.
    
    A participant is considered processed if they have output files in the
    final output directory (OUTPUT_DIR_04).
    """
    processed = set()
    if not os.path.isdir(output_dir):
        return processed
    
    pattern = re.compile(
        r"sub-(?P<sub>[A-Za-z0-9]+)_ses-01_task-experiment01_run-(?P<run>\d+)_"
        r"recording-(?P<recording>eye1|eye2|fixations|saccades)_"
        r"(?P<suffix>physio|physioevents)\.tsv\.gz$"
    )
    run_types_by_sub = {}
    for root, _, files in os.walk(output_dir):
        for file_name in files:
            match = pattern.match(file_name)
            if not match:
                continue
            sub_id = match.group("sub")
            run_idx = int(match.group("run"))
            recording = match.group("recording")
            suffix = match.group("suffix")
            key = f"{recording}:{suffix}"
            run_types_by_sub.setdefault(sub_id, {}).setdefault(run_idx, set()).add(key)

    required_runs = BLOCKS * ORDERS
    required_types = {
        "eye1:physio",
        "eye2:physio",
        "fixations:physioevents",
        "saccades:physioevents",
    }
    for sub_id, run_map in run_types_by_sub.items():
        complete = True
        for run_idx in range(1, required_runs + 1):
            if run_map.get(run_idx, set()) != required_types:
                complete = False
                break
        if complete:
            processed.add(sub_id)
    
    return processed

def get_input_participants(input_dir):
    """Get set of participants available in input directory.
    
    Participants are identified by .asc file names (without extension).
    """
    participants = set()
    if not os.path.isdir(input_dir):
        return participants
    
    for file in os.listdir(input_dir):
        if file.endswith('.asc'):
            participant_id = os.path.splitext(file)[0]
            participants.add(participant_id)
    
    return participants


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream-based eyetracking pipeline.')
    parser.add_argument('--participants', nargs='+', metavar='ID',
                        help='Only process these participant IDs (e.g. 031_01 057_01).')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if participant appears already processed.')
    parser.add_argument('--resume', action='store_true',
                        help='Like --force, but skip participants whose Phase-5 done-marker '
                             '(intermediate/<id>_analysis04.done) is newer than their pkl. '
                             'Use to continue an interrupted --force/--resume run.')
    parser.add_argument('--events-only', action='store_true',
                        help='Skip sample pixel-matching; write only fixation/saccade physioevents. '
                             'Much faster when sample CSVs already exist.')
    args = parser.parse_args()

    logger = setup_logging()

    # Results dir for the run log and processing report.
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    # Capture the full run to a timestamped log file (scaffold classes share this
    # logger, so their warnings/errors are captured too).
    run_log_path = os.path.join(results_dir, f'log_pipeline_run_{run_ts}.txt')
    _file_handler = logging.FileHandler(run_log_path, encoding='utf-8')
    _file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_file_handler)
    logger.info('Run log: %s', run_log_path)

    # Optional lab-internal participant sheet (not public): adds experimenter
    # remarks to the report. Without it the report still lists pipeline outcomes.
    sheet_path = config.PARTICIPANT_SHEET
    participant_remarks = load_participant_remarks(sheet_path, logger=logger) if sheet_path else {}

    # Per-participant outcome records, written to the processing report at the end.
    run_records = []

    def record_outcome(pid, outcome, stage, reason=''):
        run_records.append({
            'participant_id': pid,
            'session': session_of(pid),
            'outcome': outcome,
            'stage': stage,
            'reason': reason,
        })

    # Validate critical input directories
    if not os.path.isdir(INPUT_DIR):
        logger.error('Input directory not found: %s', INPUT_DIR)
        exit(1)
    if not os.path.isdir(VIDEO_DIR):
        logger.error('Video directory not found: %s', VIDEO_DIR)
        exit(1)
    if not os.path.isdir(SEGMENTATION_DIR):
        logger.error('Segmentation directory not found: %s', SEGMENTATION_DIR)
        exit(1)
    if not os.path.isfile(CATEGORY_MAPPING_FILE):
        logger.error('Category mapping file not found: %s', CATEGORY_MAPPING_FILE)
        exit(1)

    logger.info('=' * 60)
    logger.info('STREAM-BASED PIPELINE (Write-Once, Memory-Efficient)')
    logger.info('=' * 60)
    logger.info('Input (EyeLink .asc):     %s', INPUT_DIR)
    logger.info('Video directory:          %s', VIDEO_DIR)
    logger.info('Segmentation masks:       %s', SEGMENTATION_DIR)
    logger.info('Category mapping:         %s', CATEGORY_MAPPING_FILE)
    logger.info('Final output:             %s', DERIVATIVES_ROOT)
    logger.info('=' * 60)

    # Check for already-processed participants
    input_participants = get_input_participants(INPUT_DIR)
    processed_participants = get_processed_participants(DERIVATIVES_ROOT)

    if args.participants:
        requested = set(args.participants)
        unknown = requested - input_participants
        if unknown:
            logger.warning('Requested participants not found in input: %s', ', '.join(sorted(unknown)))
        input_participants &= requested

    if args.force or args.resume:
        remaining_participants = input_participants
    else:
        remaining_participants = input_participants - processed_participants

    logger.info('Total participants in input: %d', len(input_participants))
    logger.info('Already processed: %d', len(processed_participants))
    logger.info('Remaining to process: %d', len(remaining_participants))
    
    if processed_participants:
        logger.info('Already processed: %s', ', '.join(sorted(processed_participants)))
    
    if not remaining_participants:
        logger.warning('All participants have been processed. Exiting.')
        exit(0)

    # Instantiate stream-based analysis classes
    analysis00 = StreamAnalysis00(INPUT_DIR, VIDEO_DIR, MOVIE_FILES, logger=logger)
    analysis01 = StreamAnalysis01(logger=logger)
    
    video_cache = VideoMetadataCache()
    analysis02 = StreamAnalysis02(VIDEO_DIR, video_cache=video_cache, movie_files=MOVIE_FILES, logger=logger)
    
    samples_cache_dir = os.path.join(OUTPUT_DIR_04, "INTERMEDIATE_CACHE")
    analysis03 = StreamAnalysis03(
        SEGMENTATION_DIR,
        movie_files=MOVIE_FILES,
        logger=logger,
        samples_cache_dir=samples_cache_dir,
        resume_samples=True,
    )
    analysis04 = StreamAnalysis04(
        DERIVATIVES_ROOT,
        CATEGORY_MAPPING_FILE,
        logger=logger,
        task_label_ses01='experiment01',
        task_label_ses02='experiment02',
    )

    # Execute streaming pipeline per participant to limit memory usage
    logger.info('Running per-participant streaming pipeline')
    processed_count = 0
    stage03_dir = os.path.join(OUTPUT_DIR_04, "intermediate")
    os.makedirs(stage03_dir, exist_ok=True)

    for participant_id in sorted(remaining_participants):
        stage03_path = os.path.join(stage03_dir, f"{participant_id}_analysis03.pkl")
        marker_path = os.path.join(stage03_dir, f"{participant_id}_analysis04.done")

        if (args.resume and os.path.isfile(marker_path) and os.path.isfile(stage03_path)
                and os.path.getmtime(marker_path) >= os.path.getmtime(stage03_path)):
            logger.info('Resume: %s already finished Phase 5 (marker newer than pkl). Skipping.',
                        participant_id)
            record_outcome(participant_id, 'skipped', 'resume')
            continue

        # Isolate each participant: an unhandled error is logged and recorded,
        # then we move on instead of aborting the whole run.
        try:
            data_03 = None
            from_cache = False

            if os.path.isfile(stage03_path):
                try:
                    with open(stage03_path, "rb") as f:
                        data_03 = pickle.load(f)
                    from_cache = True
                    logger.info('Loaded Analysis03 intermediate from %s', stage03_path)
                except Exception as e:
                    logger.warning(
                        'Failed to load Analysis03 intermediate for %s: %s. Recomputing.',
                        participant_id,
                        e
                    )

            if data_03 is None:
                logger.info('Phase 1: Extracting raw EyeLink data for %s', participant_id)
                data_00 = analysis00.process(participant_ids=[participant_id])

                if not data_00:
                    logger.error('Analysis00 produced no data for %s. Skipping.', participant_id)
                    record_outcome(participant_id, 'skipped', 'phase1_extract',
                                   'no raw data extracted (e.g. no fixation events - see run log)')
                    continue

                logger.info('Phase 2: Chunking by movie for %s', participant_id)
                data_01 = analysis01.process(data_00)
                if not data_01:
                    logger.error('Analysis01 produced no data for %s. Skipping.', participant_id)
                    record_outcome(participant_id, 'skipped', 'phase2_chunk',
                                   'no movie chunks (e.g. screenshot session / no movie timestamps)')
                    continue

                logger.info('Phase 3: Matching frames to timestamps for %s', participant_id)
                data_02 = analysis02.process(data_01, data_00)
                if not data_02:
                    logger.error('Analysis02 produced no data for %s. Skipping.', participant_id)
                    record_outcome(participant_id, 'skipped', 'phase3_frames',
                                   'no frame matches')
                    continue

                logger.info('Phase 4: Matching to segmentation masks for %s', participant_id)
                data_03 = analysis03.process(data_02, skip_samples=args.events_only)
                if not data_03:
                    logger.error('Analysis03 produced no data for %s. Skipping.', participant_id)
                    record_outcome(participant_id, 'skipped', 'phase4_masks',
                                   'no segmentation-mask matches')
                    continue

                # Save intermediate Analysis03 output
                # Most intensive computational aspect
                try:
                    with open(stage03_path, "wb") as f:
                        pickle.dump(data_03, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info('Saved Analysis03 intermediate to %s', stage03_path)
                except Exception as e:
                    logger.error('Failed to save Analysis03 intermediate for %s: %s', participant_id, e)

            logger.info('Phase 5: Mapping to semantic categories (SAVING FINAL OUTPUT) for %s', participant_id)
            analysis04.process(data_03, participant_id_list=[participant_id], skip_samples=args.events_only)

            try:
                with open(marker_path, "w") as f:
                    f.write(f"events_only={args.events_only}\n")
                logger.info('Wrote Phase-5 done-marker to %s', marker_path)
            except Exception as e:
                logger.warning('Failed to write done-marker for %s: %s', participant_id, e)

            record_outcome(participant_id, 'completed',
                           'completed_cache' if from_cache else 'completed')
            processed_count += 1

        except Exception as e:
            logger.exception('Unhandled error processing %s: %s', participant_id, e)
            record_outcome(participant_id, 'error', 'error', str(e))
            continue

    logger.info('=' * 80)
    logger.info('Stream pipeline complete!')
    logger.info('Final output saved to: %s', DERIVATIVES_ROOT)
    logger.info('Processed %d participants', processed_count)
    logger.info('=' * 80)

    # Write the processing-issue report (cross-referenced with sheet remarks).
    try:
        write_processing_report(run_records, participant_remarks, results_dir,
                                timestamp=run_ts, logger=logger)
    except Exception as e:
        logger.error('Failed to write processing report: %s', e)
