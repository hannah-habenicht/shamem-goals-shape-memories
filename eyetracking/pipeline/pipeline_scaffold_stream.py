"""Stream-based pipeline scaffold classes (write-once, memory-efficient).

This module implements the eyetracking analysis pipeline using streaming
(in-memory) processing. Data flows through all stages and only the final
semantic category output is written to disk.

Key differences from disk-based version:
- Analysis00/01/02/03 return data dicts instead of writing to disk
- Each analysis processes data in memory and passes to next stage
- Only Analysis04 writes final output
- Processes one participant at a time through the full pipeline
- Significantly reduced disk I/O and storage footprint
"""
import os
import logging
import re
# import VALID_COORDINATE_SYSTEMS
import json
import numpy as np
import pandas as pd
import cv2
import warnings
from collections import OrderedDict
import pickle

from pipeline_utils import BLOCKS, ORDERS
from video_cache import VideoMetadataCache

# Do not let pandas hand large comparisons to numexpr. With numexpr 2.14.1
# (pandas 2.3.3, Windows) about 2% of comparisons on the ~1.3 million-row
# sample table returned wrong results, flipping large contiguous blocks of
# rows. That silently emptied or truncated sample chunks for random viewings.
# Fixation and saccade tables are below pandas' numexpr threshold (1e6 rows)
# and were never affected. Plain numpy evaluation is exact.
pd.set_option('compute.use_numexpr', False)

# Sample-count check in StreamAnalysis01: a chunk should contain about one
# sample per 2 ms of viewing (500 Hz). Fewer than this share is reported.
MIN_SAMPLE_COVERAGE = 0.95

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


class StreamAnalysis00:
    """Extract raw EyeLink data from .asc files (returns in-memory data).
    
    Instead of saving to disk, returns a dict with parsed eyetracking events.
    """
    
    def __init__(self, input_dir, video_dir, movie_files, logger=None, video_cache=None):
        self.input_dir = input_dir
        self.video_dir = video_dir
        self.movie_files = movie_files or {}
        self.logger = logger or logging.getLogger('StreamAnalysis00')
        self.video_cache = video_cache

    def process(self, participant_ids=None):
        """Parse all .asc files and return data in memory.
        
        Returns:
            dict: {file_name: {
                'fixations': DataFrame,
                'saccades': DataFrame,
                'samples': DataFrame,
                'movie_timestamps': DataFrame
            }}
        """
        result = {}
        self.logger.info('STARTING StreamAnalysis00 - extracting raw EyeLink data')

        if not os.path.isdir(self.input_dir):
            self.logger.error('Input directory not found: %s', self.input_dir)
            return result

        for file in os.listdir(self.input_dir):
            if not file.endswith('.asc'):
                continue

            input_file = os.path.join(self.input_dir, file)
            EL_FileStart = os.path.splitext(file)[0]
            if participant_ids and EL_FileStart not in participant_ids:
                continue
            self.logger.info('Processing file: %s', EL_FileStart)

            # Read file
            with open(input_file, 'r') as f:
                fileTxt0 = f.read().splitlines(True)                # split into lines
            fileTxt0 = list(filter(None, fileTxt0))                 # remove emptys
            fileTxt0 = np.array(fileTxt0)                           # convert to np array for simpler indexing

            # Separate lines into samples and messagesa
            nLines = len(fileTxt0)
            lineType = np.array(['default'] * nLines, dtype='object') # Placeholder
            

            iStartRec = None
            for i in range(nLines):
                if len(fileTxt0[i]) < 3:
                    lineType[i] = 'EMPTY'
                elif fileTxt0[i].startswith('*') or fileTxt0[i].startswith('>>>>>'):
                    lineType[i] = 'COMMENT'
                elif fileTxt0[i].split()[0][0].isdigit() or fileTxt0[i].split()[0].startswith('-'):
                    lineType[i] = 'SAMPLE'
                else:
                    lineType[i] = fileTxt0[i].split()[0]
                if '!CAL' in fileTxt0[i]:
                    iStartRec = i + 1
            
            if iStartRec is None:
                iStartRec = 0

            # Parse events

            # Read all lines of one event type, returning an empty DataFrame (with
            # the expected columns) when that event never occurs in the file.
            # Without this, pd.read_csv is handed a skiprows list covering every
            # line and raises EmptyDataError (e.g. a recording with no EFIX
            # fixations, as seen in 039_02).
            def read_event(event_type, usecols, columns):
                iNot = np.nonzero(lineType != event_type)[0]
                if len(iNot) >= nLines:
                    return pd.DataFrame(columns=columns)
                df = pd.read_csv(input_file, skiprows=iNot.tolist(), header=None,
                                 sep=r"\s+", usecols=usecols)
                df.columns = columns
                return df

            # Find Start indexes
            dfRecStart = read_event('START', [1], ['tStart'])

            # Find End indexes
            dfRecEnd = read_event('END', [1, 5, 6], ['tEnd', 'xRes', 'yRes'])

            # Combine start and end into table
            dfTrials = pd.concat([dfRecStart, dfRecEnd], axis=1)

            # Messages
            iMsg = np.nonzero(lineType == 'MSG')[0]
            tMsg = []
            txtMsg = []
            for i in range(len(iMsg)):
                info = fileTxt0[iMsg[i]].split()    # separate MSG prefix and timestamp from rest of message 
                tMsg.append(int(info[1]))           # extract info
                txtMsg.append(' '.join(info[2:])) 
            dfMsg = pd.DataFrame({'time': tMsg, 'text': txtMsg})

            # Fixations
            dfFix = read_event('EFIX', range(1, 8),
                               ['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg'])

            # Saccades
            dfSacc = read_event('ESACC', range(1, 11),
                                ['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart',
                                 'xEnd', 'yEnd', 'ampDeg', 'vPeak'])

            # Blinks
            dfBlink = read_event('EBLINK', range(1, 5),
                                 ['eye', 'tStart', 'tEnd', 'duration'])

            # A recording with no fixations cannot yield usable output (and the
            # eye/sample-column inference below relies on dfFix). Skip it loudly
            # rather than crashing the whole run.
            if dfFix.empty:
                self.logger.warning(
                    'File %s has no EFIX fixation events; skipping (unusable for this pipeline).',
                    EL_FileStart)
                continue

            #### Samples ####
            # determine sample columns based on eyes recorded in file
            eyesInFile = np.unique(dfFix.eye) if not dfFix.empty else np.array([]) #Raise error, empty
            if len(eyesInFile) > 1:
                print('binocular data detected.')
                cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
            else:
                eye = eyesInFile[0]
                print(f'monocular data detected ({eye} eye).')
                cols = ['tSample', f'{eye}X', f'{eye}Y', f'{eye}Pupil']

            # Import samples
            if iStartRec is not None:
                # iNotSample is all lines that are NOT! SAMPLE, plus any sample lines before iStartRec (ie outside of recording window)
                iNotSample = np.nonzero(np.logical_or(lineType != 'SAMPLE', np.arange(nLines) < iStartRec))[0]
            else:
                iNotSample = np.nonzero(lineType != 'SAMPLE')[0]
            
            if len(iNotSample) == nLines:           #no samples safeguard
                raise EOFError("No SAMPLE lines found in file")
            
            else:
                dfSamples = pd.read_csv(input_file, skiprows=iNotSample.tolist(), header=None, sep=r"\s+", usecols=range(0, len(cols)))
                dfSamples.columns = cols


            # Normalize L/R columns
            for eye in ['L', 'R']:
                for suf in ('X', 'Y', 'Pupil'):
                    col = f'{eye}{suf}'
                    if col not in dfSamples.columns:
                        dfSamples[col] = np.nan
                    else:
                        dfSamples[col] = pd.to_numeric(dfSamples[col], errors='coerce')

            # Extract movie timestamps (if *01.asc file)
            dfmovie_timestamps = pd.DataFrame()
            
            if file.endswith('01.asc') and not dfMsg.empty: #
                # #Extract MSG details that pertain to the Movies
                dfcontain_movies = dfMsg[dfMsg['text'].str.contains('movie')].astype({"text": str})
                
                if not dfcontain_movies.empty:
                    dfcontain_movies['movie'] = dfcontain_movies['text'].str[-20:]
                    dfcontain_movies['val'] = dfcontain_movies['text'].str[:4]
                    dfcontain_movies['movie'] = dfcontain_movies['movie'].str.replace('mp4.', '', regex=False)
                    dfcontain_movies = dfcontain_movies.drop(['text'], axis=1)

                    # ===== CLEAN UP MOVIE TIME STAMPS FILE  ===== #
                    # Pair each "started" message with the next "stopped" message
                    # of the same movie. Pairing the two lists by position breaks
                    # when a stop message is missing: some sessions end without a
                    # final "stopped" message (and without an END record), which
                    # left the last viewing without an end time so it was silently
                    # dropped. A missing end is filled from the film length below.
                    starts, ends, nums = [], [], []
                    open_idx = None
                    for t_msg, mv in zip(dfcontain_movies['time'], dfcontain_movies['movie']):
                        num = mv[:8]
                        if 'started' in mv:
                            if open_idx is not None:
                                self.logger.warning('%s: %s started while %s had no stop message',
                                                    EL_FileStart, num, nums[open_idx])
                            starts.append(t_msg)
                            ends.append(np.nan)
                            nums.append(num)
                            open_idx = len(starts) - 1
                        elif 'stopped' in mv:
                            if open_idx is not None and nums[open_idx] == num:
                                ends[open_idx] = t_msg
                                open_idx = None
                            else:
                                self.logger.warning('%s: stop message for %s without a matching start; ignored',
                                                    EL_FileStart, num)
                    dft_start = pd.DataFrame({'t_start(ms)': starts})
                    dft_end = pd.DataFrame({'t_end(ms)': ends})
                    dfmovie_num = pd.DataFrame({'movie_num': nums})
                    
                    ORDER = 1
                    BLOCK = 1
                    df_block = pd.DataFrame(columns=['Block'])
                    df_order = pd.DataFrame(columns=['Order'])
                    dft_diff = pd.DataFrame(columns=['t_diff(ms)'])

                    for index, row in dfmovie_num.iterrows():
                        movie_num = row['movie_num']
                        video_file = self.movie_files.get(movie_num)
                        if not video_file:
                            self.logger.warning('No video mapping for movie %s', movie_num)
                            break

                        video_path = os.path.join(self.video_dir, video_file)
                        try:
                            if self.video_cache is not None:
                                info = self.video_cache.get(video_path)
                                length_ms = int((info['total_frames'] / info['frame_rate']) * 1000)
                            else:
                                cap = cv2.VideoCapture(video_path)
                                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                                if frame_rate == 0:
                                    self.logger.error('Could not retrieve frame rate for video %s', video_file)
                                    cap.release()
                                    break
                                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                length_ms = int((total_frames / frame_rate) * 1000)
                                cap.release()
                        except Exception as e:
                            self.logger.error('Error reading video %s: %s', video_path, e)
                            break

                        row_idx = int(index)
                        dft_diff.loc[row_idx, 't_diff(ms)'] = length_ms
                        df_block.loc[row_idx, 'Block'] = int(BLOCK)
                        df_order.loc[row_idx, 'Order'] = int(ORDER)

                        ORDER += 1
                        if ORDER > ORDERS:
                            ORDER = 1
                            BLOCK += 1
                            if BLOCK > BLOCKS:
                                BLOCK = 1

                    dfmovie_timestamps = pd.concat(
                        [dft_start, dft_end, dfmovie_num, dft_diff, df_block, df_order],
                        axis=1
                    ).reset_index(drop=True)
                    dfmovie_timestamps['Block'] = pd.to_numeric(dfmovie_timestamps['Block'], errors='coerce')
                    dfmovie_timestamps['Order'] = pd.to_numeric(dfmovie_timestamps['Order'], errors='coerce')

                    # Viewings without a stop message end at start + film length.
                    no_end = dfmovie_timestamps['t_end(ms)'].isna()
                    if no_end.any():
                        length = pd.to_numeric(dfmovie_timestamps['t_diff(ms)'], errors='coerce')
                        dfmovie_timestamps.loc[no_end, 't_end(ms)'] = (
                            dfmovie_timestamps.loc[no_end, 't_start(ms)'] + length[no_end])
                        for _, r in dfmovie_timestamps[no_end].iterrows():
                            self.logger.warning(
                                '%s: no stop message for %s (Block %s, Order %s); end set to start + film length',
                                EL_FileStart, r['movie_num'], r['Block'], r['Order'])

            result[EL_FileStart] = {
                'trials': dfTrials,
                'messages': dfMsg,
                'fixations': dfFix,
                'saccades': dfSacc,
                'blinks': dfBlink,
                'samples': dfSamples,
                'movie_timestamps': dfmovie_timestamps
            }
            # self.logger.info('Result keys after %s: %s', EL_FileStart, list(result.keys()))

        self.logger.info('StreamAnalysis00 complete')
        return result

class StreamAnalysis01:
    """Chunk eyetracking data by movie timeframe (memory-based)."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('StreamAnalysis01')

    def process(self, data_from_00):
        """chunks samples/fixations/saccades into per-movie blocks
        
        Args:
            data_from_00: Output from StreamAnalysis00.process()
        
        Returns:
            dict: {participant: {block_order: {
                'samples': DataFrame,
                'fixations': DataFrame,
                'saccades': DataFrame
            }}}
        """
        result = {}
        self.logger.info('STARTING StreamAnalysis01 - chunking samples per movie')

        for participant_id, analysis_data in data_from_00.items():
            if analysis_data['movie_timestamps'].empty:
                self.logger.info('No movie timestamps for %s; skipping chunking', participant_id)
                continue

            df_ts_movie = analysis_data['movie_timestamps']
            df_samples = analysis_data['samples']
            df_fixations = analysis_data['fixations']
            df_saccades = analysis_data['saccades']

            chunks = {}
            start_time  = df_ts_movie['t_start(ms)']
            end_time    = df_ts_movie['t_end(ms)']

            for BLOCK in range(1, BLOCKS + 1):
                for ORDER in range(1, ORDERS + 1):
                    row = (BLOCK - 1) * ORDERS + (ORDER - 1)
                    if row >= len(df_ts_movie):
                        break

                    movie_num = df_ts_movie['movie_num'].iloc[row]
                    
                    # Extract Samples Data Chunk
                    chunk_samples = df_samples[(df_samples['tSample'] >= start_time.iloc[row]) & 
                                              (df_samples['tSample'] <= end_time.iloc[row])].copy()
                    chunk_samples['Movie'] = movie_num

                    # Extract Fixations Data Chunk
                    chunk_fixations = df_fixations[(df_fixations['tStart'] >= start_time.iloc[row]) & 
                                                  (df_fixations['tEnd'] <= end_time.iloc[row])].copy()
                    chunk_fixations['Movie'] = movie_num
                    
                    # Extract Saccades Data Chunk
                    chunk_saccades = df_saccades[(df_saccades['tStart'] >= start_time.iloc[row]) & 
                                                (df_saccades['tEnd'] <= end_time.iloc[row])].copy()
                    chunk_saccades['Movie'] = movie_num

                    # Skip if all empty
                    if chunk_samples.empty and chunk_fixations.empty and chunk_saccades.empty:
                        continue

                    # Report chunks whose sample count does not match their
                    # duration, instead of passing a short chunk on silently.
                    expected = (end_time.iloc[row] - start_time.iloc[row]) / 2.0 + 1
                    if expected > 0 and len(chunk_samples) < MIN_SAMPLE_COVERAGE * expected:
                        self.logger.warning(
                            '%s Block_%d_Order_%d_%s: %d samples, expected about %d (500 Hz); '
                            'samples may be missing',
                            participant_id, BLOCK, ORDER, movie_num, len(chunk_samples), int(expected))

                    block_order_key = f'Block_{BLOCK}_Order_{ORDER}_{movie_num}'
                    chunks[block_order_key] = {
                        'samples': chunk_samples,
                        'fixations': chunk_fixations,
                        'saccades': chunk_saccades
                    }

            result[participant_id] = chunks

        self.logger.info('StreamAnalysis01 complete')
        return result

class StreamAnalysis02:
    """Match chunked data to video frames (memory-based)."""
    
    def __init__(self, video_dir, video_cache=None, movie_files=None, logger=None):
        self.video_dir = video_dir
        self.video_cache = video_cache or VideoMetadataCache()
        self.movie_files = movie_files or {}
        self.logger = logger or logging.getLogger('StreamAnalysis02')

    @staticmethod
    def _nearest_frame_indices(frame_timestamps, query_times):
        """Vectorized nearest-frame lookup for sorted frame_timestamps (ms)."""
        if len(frame_timestamps) == 0:
            return np.array([], dtype=int)

        frame_ts = np.asarray(frame_timestamps, dtype=float)
        q = np.asarray(query_times, dtype=float)

        idx = np.searchsorted(frame_ts, q, side='left')
        idx_right = np.clip(idx, 0, len(frame_ts) - 1)
        idx_left = np.clip(idx - 1, 0, len(frame_ts) - 1)

        dist_left = np.abs(q - frame_ts[idx_left])
        dist_right = np.abs(q - frame_ts[idx_right])
        choose_left = dist_left <= dist_right

        return np.where(choose_left, idx_left, idx_right).astype(int)

    def process(self, data_from_01, movie_timestamps_from_00):
        """Add matched frame indices to eyetracking data.
        
        Args:
            data_from_01: Chunked data from StreamAnalysis01
            movie_timestamps_from_00: Movie timestamps from StreamAnalysis00
        
        Returns:
            dict: Same structure as input with added 'matched_frame_IDx' columns
        """
        self.logger.info('STARTING StreamAnalysis02 - matching frames to timestamps')

        result = {}

        for participant_id, chunks in data_from_01.items():
            if participant_id not in movie_timestamps_from_00:
                self.logger.warning('No movie timestamps for %s', participant_id)
                continue

            df_ts_movie = movie_timestamps_from_00[participant_id]['movie_timestamps']
            
            matched_chunks = {}
            for chunk_key, chunk_data in chunks.items():
                matched_chunk = {}

                for eyetracking_type in ['samples', 'fixations', 'saccades']:
                    df = chunk_data[eyetracking_type]
                    if df.empty:
                        matched_chunk[eyetracking_type] = df
                        continue

                    movie_label = df['Movie'].iloc[0]
                    video_file = self.movie_files.get(movie_label)
                    if not video_file:
                        self.logger.warning('No video mapping for movie %s', movie_label)
                        matched_chunk[eyetracking_type] = df
                        continue

                    video_path = video_file if os.path.isabs(video_file) else os.path.join(self.video_dir or '', video_file)

                    try:
                        info = self.video_cache.get(video_path)
                        total_frames = info['total_frames']
                        frame_timestamps = info['frame_timestamps']
                    except Exception as e:
                        self.logger.warning('Could not read video %s: %s', video_path, e)
                        matched_chunk[eyetracking_type] = df
                        continue

                    # Extract block/order from chunk key
                    parts = chunk_key.split('_')
                    block_num = int(parts[1])
                    order_num = int(parts[3])
                    
                    # self.logger.info("df_ts_movie columns: %s", df_ts_movie.columns)
                    # self.logger.info("df_ts_movie dtypes: %s", df_ts_movie.dtypes)
                    # self.logger.info("df_ts_movie head:\n%s", df_ts_movie.head())


                    # Get timing reference | # find movie starting time based on BLOCK and ORDER number. Cant use movie_num, as the names repeat
                    try:    
                        t_start_movie = float(df_ts_movie.loc[(df_ts_movie['Block'] == block_num) & 
                                                             (df_ts_movie['Order'] == order_num), 
                                                             't_start(ms)'].iloc[0])
                    except:
                        self.logger.warning('Could not find movie timing for Block %d Order %d', block_num, order_num)
                        matched_chunk[eyetracking_type] = df
                        continue

                    df_matched = df.copy()

                    if eyetracking_type == 'samples':
                        abs_times = df['tSample'].to_numpy(dtype=float) - t_start_movie
                        matched_frames = self._nearest_frame_indices(frame_timestamps, abs_times)
                        df_matched['matched_frame_IDx'] = matched_frames
                    else:  # fixations or saccades
                        abs_times_start = df['tStart'].to_numpy(dtype=float) - t_start_movie
                        abs_times_end = df['tEnd'].to_numpy(dtype=float) - t_start_movie
                        matched_frames_start = self._nearest_frame_indices(frame_timestamps, abs_times_start)
                        matched_frames_end = self._nearest_frame_indices(frame_timestamps, abs_times_end)
                        df_matched['matched_frame_IDx_start'] = matched_frames_start
                        df_matched['matched_frame_IDx_end'] = matched_frames_end

                    matched_chunk[eyetracking_type] = df_matched

                matched_chunks[chunk_key] = matched_chunk

            result[participant_id] = matched_chunks

        self.logger.info('StreamAnalysis02 complete')
        return result

class StreamAnalysis03:
    """Match eyetracking data to semantic segmentation masks (memory-based)."""
    
    def __init__(self, segmentation_dir, movie_files=None, logger=None,
                 frame_width=1920, frame_height=1080, cache_max_frames=128,
                 samples_cache_dir=None, resume_samples=True):
        self.segmentation_dir = segmentation_dir
        self.movie_files = movie_files or {}
        self.logger = logger or logging.getLogger('StreamAnalysis03')
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cache_max_frames = int(cache_max_frames) if cache_max_frames else 0
        self._frame_cache = OrderedDict()
        self.samples_cache_dir = samples_cache_dir
        self.resume_samples = bool(resume_samples)
        if self.samples_cache_dir:
            os.makedirs(self.samples_cache_dir, exist_ok=True)

    def process(self, data_from_02, skip_samples=False):
        """Extract pixel category indices from segmentation masks.

        Args:
            data_from_02: Matched frames data from StreamAnalysis02
            skip_samples: If True, pass samples through unprocessed (much faster
                when only fixations/saccades are needed).

        Returns:
            dict: Same structure with added 'MatchingPixel' columns
        """
        self.logger.info('STARTING StreamAnalysis03 - segmentation mask matching')
        if skip_samples:
            self.logger.info('skip_samples=True: sample pixel matching will be skipped')

        result = {}

        for participant_id, chunks in data_from_02.items():
            segmented_chunks = {}

            for chunk_key, chunk_data in chunks.items():
                segmented_chunk = {}

                for eyetracking_type in ['fixations', 'saccades', 'samples',]:
                    df = chunk_data[eyetracking_type]
                    if df.empty:
                        segmented_chunk[eyetracking_type] = df
                        continue

                    if skip_samples and eyetracking_type == 'samples':
                        segmented_chunk[eyetracking_type] = df
                        continue

                    cache_path = None
                    if eyetracking_type == 'samples':
                        if self.samples_cache_dir:
                            safe_participant = str(participant_id).replace(os.sep, '_').replace(':', '_')
                            safe_chunk = str(chunk_key).replace(os.sep, '_').replace(':', '_')
                            cache_path = os.path.join(
                                self.samples_cache_dir,
                                f"{safe_participant}_{safe_chunk}_samples.pkl"
                            )
                        if cache_path and self.resume_samples and os.path.isfile(cache_path):
                            try:
                                with open(cache_path, "rb") as f:
                                    df_seg_cached = pickle.load(f)
                                segmented_chunk[eyetracking_type] = df_seg_cached
                                self.logger.info("Loaded cached samples for %s / %s", participant_id, chunk_key)
                                continue
                            except Exception:
                                self.logger.exception("Failed to load cached samples: %s", cache_path)

                    movie_label = df['Movie'].iloc[0]
                    mask_folder = self._find_mask_folder(movie_label)
                    if not mask_folder:
                        self.logger.warning('No mask folder for movie %s', movie_label)
                        segmented_chunk[eyetracking_type] = df
                        continue

                    df_seg = df.copy()
                    total_frames = len(os.listdir(mask_folder))

                    if eyetracking_type == 'samples':
                        df_seg['MatchingPixel'] = -1
                        df_seg['OutOfBounds'] = (
                            (df_seg['LX'] < 0) | (df_seg['LX'] >= self.frame_width) |
                            (df_seg['LY'] < 0) | (df_seg['LY'] >= self.frame_height) |
                            (df_seg['RX'] < 0) | (df_seg['RX'] >= self.frame_width) |
                            (df_seg['RY'] < 0) | (df_seg['RY'] >= self.frame_height)
                        )

                        lx = df_seg['LX'].to_numpy()
                        ly = df_seg['LY'].to_numpy()
                        rx = df_seg['RX'].to_numpy()
                        ry = df_seg['RY'].to_numpy()
                        frame_idx_arr = df_seg['matched_frame_IDx'].to_numpy()

                        # Build validity masks (vectorized)
                        left_ok = pd.notna(lx) & pd.notna(ly)
                        right_ok = pd.notna(rx) & pd.notna(ry)
                        in_bounds = ~df_seg['OutOfBounds'].to_numpy()
                        valid_frame = (frame_idx_arr >= 0) & (frame_idx_arr < (total_frames / 2))
                        valid = in_bounds & valid_frame & (left_ok | right_ok)

                        valid_frames = frame_idx_arr[valid]
                        for frame_idx in np.unique(valid_frames):
                            frame_data = self._get_frame_data(mask_folder, int(frame_idx))
                            if frame_data is None:
                                raise FileNotFoundError(f"Frame data not found for {mask_folder}, frame {frame_idx}")

                            idxs = np.where(valid & (frame_idx_arr == frame_idx))[0]
                            if idxs.size == 0:
                                continue

                            left_idxs = idxs[left_ok[idxs]]
                            if left_idxs.size:
                                labels_left = frame_data[ly[left_idxs].astype(int), lx[left_idxs].astype(int)]
                                df_seg.iloc[left_idxs, df_seg.columns.get_loc('MatchingPixel')] = labels_left

                            right_idxs = idxs[~left_ok[idxs] & right_ok[idxs]]
                            if right_idxs.size:
                                labels_right = frame_data[ry[right_idxs].astype(int), rx[right_idxs].astype(int)]
                                df_seg.iloc[right_idxs, df_seg.columns.get_loc('MatchingPixel')] = labels_right

                        if cache_path:
                            try:
                                with open(cache_path, "wb") as f:
                                    pickle.dump(df_seg, f, protocol=pickle.HIGHEST_PROTOCOL)
                                self.logger.info("Saved cached samples for %s / %s", participant_id, chunk_key)
                            except Exception:
                                self.logger.exception("Failed to save cached samples: %s", cache_path)
                    else:  
                        # fixations or saccades
                        if eyetracking_type == 'fixations':
                            df_seg['MatchingPixel_Start'] = -1
                            df_seg['MatchingPixel_End'] = -1
                            x_col, y_col = 'xAvg', 'yAvg'
                        else:  # saccades
                            df_seg['MatchingPixel_Start'] = -1
                            df_seg['MatchingPixel_End'] = -1
                            x_col, y_col = 'xStart', 'yStart'
                            x_end_col, y_end_col = 'xEnd', 'yEnd'

                        # Ensure numeric types for coordinate comparisons
                        df_seg[x_col] = pd.to_numeric(df_seg[x_col], errors='coerce')
                        df_seg[y_col] = pd.to_numeric(df_seg[y_col], errors='coerce')
                        if eyetracking_type == 'saccades':
                            df_seg[x_end_col] = pd.to_numeric(df_seg[x_end_col], errors='coerce')
                            df_seg[y_end_col] = pd.to_numeric(df_seg[y_end_col], errors='coerce')

                        df_seg['OutOfBounds'] = (
                            (df_seg[x_col] < 0) | (df_seg[x_col] >= self.frame_width) |
                            (df_seg[y_col] < 0) | (df_seg[y_col] >= self.frame_height)
                        )
                        if eyetracking_type == 'saccades':
                            df_seg['OutOfBounds'] = df_seg['OutOfBounds'] | (
                                (df_seg[x_end_col] < 0) | (df_seg[x_end_col] >= self.frame_width) |
                                (df_seg[y_end_col] < 0) | (df_seg[y_end_col] >= self.frame_height)
                            )

                        x_start = df_seg[x_col].to_numpy()
                        y_start = df_seg[y_col].to_numpy()
                        if eyetracking_type == 'saccades':
                            x_end = df_seg[x_end_col].to_numpy()
                            y_end = df_seg[y_end_col].to_numpy()
                        else:
                            x_end = x_start
                            y_end = y_start

                        frame_start_arr = df_seg['matched_frame_IDx_start'].to_numpy()
                        frame_end_arr = df_seg['matched_frame_IDx_end'].to_numpy()

                        in_bounds = ~df_seg['OutOfBounds'].to_numpy()
                        valid_frame_start = (frame_start_arr >= 0) & (frame_start_arr < (total_frames / 2))
                        valid_frame_end = (frame_end_arr >= 0) & (frame_end_arr < (total_frames / 2))

                        valid_start = in_bounds & valid_frame_start & pd.notna(x_start) & pd.notna(y_start)
                        valid_end = in_bounds & valid_frame_end & pd.notna(x_end) & pd.notna(y_end)

                        for frame_idx in np.unique(frame_start_arr[valid_start]):
                            frame_data = self._get_frame_data(mask_folder, int(frame_idx))
                            if frame_data is None:
                                raise FileNotFoundError(f"Frame data not found for {mask_folder}, frame {frame_idx}")

                            idxs = np.where(valid_start & (frame_start_arr == frame_idx))[0]
                            if idxs.size == 0:
                                continue

                            labels_start = frame_data[y_start[idxs].astype(int), x_start[idxs].astype(int)]
                            df_seg.iloc[idxs, df_seg.columns.get_loc('MatchingPixel_Start')] = labels_start

                        for frame_idx in np.unique(frame_end_arr[valid_end]):
                            frame_data = self._get_frame_data(mask_folder, int(frame_idx))
                            if frame_data is None:
                                raise FileNotFoundError(f"Frame data not found for {mask_folder}, frame {frame_idx}")

                            idxs = np.where(valid_end & (frame_end_arr == frame_idx))[0]
                            if idxs.size == 0:
                                continue

                            labels_end = frame_data[y_end[idxs].astype(int), x_end[idxs].astype(int)]
                            df_seg.iloc[idxs, df_seg.columns.get_loc('MatchingPixel_End')] = labels_end

                    segmented_chunk[eyetracking_type] = df_seg

                segmented_chunks[chunk_key] = segmented_chunk

            result[participant_id] = segmented_chunks

        self.logger.info('StreamAnalysis03 complete')
        return result

    def _get_frame_data(self, mask_folder, frame_index):
        """Load frame data with caching."""
        cache_key = (mask_folder, frame_index)
        if cache_key in self._frame_cache:
            frame_data = self._frame_cache.pop(cache_key)
            self._frame_cache[cache_key] = frame_data
            return frame_data

        if cache_key not in self._frame_cache:
            base_name = os.path.basename(mask_folder)

            # Prefer pre-converted grayscale label PNGs (see convert_masks_to_labels.py);
            # they decode ~20-35x faster than parsing the original CSVs.
            labels_folder = os.path.join(os.path.dirname(mask_folder) + '_labels', base_name)
            labels_path = os.path.join(labels_folder, f"{base_name}_frame_{frame_index:04d}.png")
            frame_arr = None
            if os.path.exists(labels_path):
                frame_arr = cv2.imread(labels_path, cv2.IMREAD_UNCHANGED)
                if frame_arr is None:
                    self.logger.warning("Failed to decode label PNG, falling back to CSV: %s", labels_path)

            if frame_arr is None:
                frame_filename = f"{base_name}_frame_{frame_index:04d}.csv"
                frame_path = os.path.join(mask_folder, frame_filename)

                ## add check, validation pathway exists, file exists, and can be read before trying to read it. If not, log a warning and return None
                if not os.path.exists(mask_folder):
                    self.logger.warning(f"Mask folder does not exist: {mask_folder}")
                    return None
                if not os.path.exists(frame_path):
                    self.logger.warning(f"Frame file does not exist: {frame_path}")
                    return None
                try:
                    # header=0: the first line of these CSVs is a column-index header
                    # (';0;1;2;...'); reading it with header=None shifts every mask
                    # row down by one and corrupts y=0 lookups.
                    frame_data = pd.read_csv(frame_path, delimiter=';', header=0, index_col=0)
                    frame_arr = frame_data.to_numpy()
                except Exception:
                    self.logger.exception("Failed to read frame file: %s", frame_path)
                    return None

            self._frame_cache[cache_key] = frame_arr
            if self.cache_max_frames > 0 and len(self._frame_cache) > self.cache_max_frames:
                self._frame_cache.popitem(last=False)
        return self._frame_cache.get(cache_key)

    def _find_mask_folder(self, movie_name):
        """Find mask folder for a movie."""
        if movie_name not in self.movie_files:
            return None
        movie_file = self.movie_files[movie_name]
        movie_prefix = os.path.splitext(os.path.basename(movie_file))[0]
        for folder in os.listdir(self.segmentation_dir):
            if folder.startswith(movie_prefix):
                return os.path.join(self.segmentation_dir, folder)
        return None

class StreamAnalysis04:
    """Map pixel indices to semantic categories and save strict-BIDS derivatives."""

    def __init__(self, output_dir, category_mapping_file, logger=None,
                 task_label_ses01='experiment01', task_label_ses02='experiment02',
                 skip_participants_tsv=False):
        self.output_dir = output_dir
        self.category_mapping_file = category_mapping_file
        self.logger = logger or logging.getLogger('StreamAnalysis04')
        self.task_label_ses01 = task_label_ses01
        self.task_label_ses02 = task_label_ses02
        self.skip_participants_tsv = skip_participants_tsv
        self.category_mapping = self._load_category_mapping()
        self._ensure_derivatives_metadata()

    @staticmethod
    def _sanitize_bids_label(text):
        value = re.sub(r'[^A-Za-z0-9]', '', str(text))
        return value or 'unknown'

    @classmethod
    def parse_participant_session(cls, participant_id):
        participant_text = str(participant_id)
        if participant_text.endswith('_02'):
            sub_id = participant_text[:-3]
            session = '02'
        elif participant_text.endswith('_01'):
            sub_id = participant_text[:-3]
            session = '01'
        else:
            sub_id = participant_text
            session = '01'
        return cls._sanitize_bids_label(sub_id), session

    @staticmethod
    def _parse_run_from_chunk_key(chunk_key):
        text = str(chunk_key)
        block_match = re.search(r'Block_(\d+)_Order_(\d+)', text)
        if block_match:
            block = int(block_match.group(1))
            order = int(block_match.group(2))
            run_index = ((block - 1) * ORDERS) + order
            return run_index, block, order
        trial_match = re.search(r'Trial_(\d+)', text)
        if trial_match:
            return int(trial_match.group(1)), None, None
        return 0, None, None

    def _build_base_entities(self, participant_id, chunk_key):
        sub_id, session = self.parse_participant_session(participant_id)
        task_label = self.task_label_ses02 if session == '02' else self.task_label_ses01
        run_index, block, order = self._parse_run_from_chunk_key(chunk_key)

        run_label = f'{run_index:02d}' if run_index > 0 else '00'

        out_dir = os.path.join(self.output_dir, f'sub-{sub_id}', f'ses-{session}', 'eyetrack')
        os.makedirs(out_dir, exist_ok=True)

        base_stem = f'sub-{sub_id}_ses-{session}_task-{task_label}_run-{run_label}'
        return out_dir, base_stem, run_index, block, order

    def _ensure_derivatives_metadata(self):
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'dataset_description.json')
        if os.path.isfile(output_path):
            return
        payload = {
            'Name': 'Eyetracking Pipeline Derivatives',
            'BIDSVersion': '1.11.1',
            'DatasetType': 'derivative',
            'GeneratedBy': [
                {
                    'Name': 'eyetracking-pipeline',
                    'Version': '1.0.0',
                    'Description': 'Stream pipeline mapping eyetracking to semantic labels.',
                }
            ],
        }
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception:
            self.logger.exception('Failed writing dataset_description.json')

    def _update_participants_tsv(self, sub_id):
        tsv_path = os.path.join(self.output_dir, 'participants.tsv')
        label = f'sub-{sub_id}'
        if os.path.isfile(tsv_path):
            try:
                existing = pd.read_csv(tsv_path, sep='\t', dtype=str)
            except Exception:
                existing = pd.DataFrame(columns=['participant_id'])
        else:
            existing = pd.DataFrame(columns=['participant_id'])
        if label not in existing['participant_id'].values:
            existing = pd.concat(
                [existing, pd.DataFrame([{'participant_id': label}])],
                ignore_index=True,
            )
            existing.to_csv(tsv_path, sep='\t', index=False)

    @staticmethod
    def _write_tsv_gz(df, output_path):
        df.to_csv(output_path, sep='\t', index=False, header=False, compression='gzip')

    @staticmethod
    def _write_json(output_path, payload):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _infer_sampling_frequency(df):
        if 'tSample' not in df.columns or len(df) < 2:
            return 1000.0
        diffs = pd.to_numeric(df['tSample'], errors='coerce').diff().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 1000.0
        median_ms = float(diffs.median())
        if median_ms <= 0:
            return 1000.0
        return round(1000.0 / median_ms, 6)

    @staticmethod
    def _sample_eye_dataframe(df, eye_prefix):
        x_col = f'{eye_prefix}X'
        y_col = f'{eye_prefix}Y'
        pupil_col = f'{eye_prefix}Pupil'
        return pd.DataFrame({
            'timestamp': pd.to_numeric(df.get('tSample', np.nan), errors='coerce'),
            'x_coordinate': pd.to_numeric(df.get(x_col, np.nan), errors='coerce'),
            'y_coordinate': pd.to_numeric(df.get(y_col, np.nan), errors='coerce'),
            'pupil_size': pd.to_numeric(df.get(pupil_col, np.nan), errors='coerce'),
            'matching_pixel': df.get('MatchingPixel', -1),
            'out_of_bounds': df.get('OutOfBounds', False),
            'assigned_category': df.get('AssignedCategory', ''),
            'category': df.get('Category', ''),
            'designation': df.get('Designation', ''),
            'source_chunk_key': df.get('source_chunk_key', ''),
            'run_index': df.get('run_index', 0),
            'block': df.get('block', np.nan),
            'order': df.get('order', np.nan),
        })

    @staticmethod
    def _physioevents_dataframe(df):
        preferred = ['onset']
        if 'duration' in df.columns:
            preferred.append('duration')
        remaining = [col for col in df.columns if col not in preferred]
        return df[preferred + remaining]

    def _load_category_mapping(self):
        try:
            # index_col=1: key by the Idx column (1-indexed ADE20K class ids, matching
            # the pred+1 values in the mask CSVs). index_col=0 keyed by the 0-based row
            # counter, shifting every lookup by one class.
            df = pd.read_excel(self.category_mapping_file, index_col=1)
            mapping = {}
            for idx, row in df.iterrows():
                mapping[idx] = (
                    row.get('Name', ''),
                    row.get('Category', ''),
                    row.get('Designation', '')
                )
            self.logger.info('Loaded category mapping with %d entries', len(mapping))
            return mapping
        except Exception as e:
            self.logger.error('Failed to load category mapping: %s', e)
            return {}

    def process(self, data_from_03, participant_id_list=None, skip_samples=False):
        self.logger.info('STARTING StreamAnalysis04 - semantic category mapping')
        os.makedirs(self.output_dir, exist_ok=True)

        for participant_id, chunks in data_from_03.items():
            if participant_id_list and participant_id not in participant_id_list:
                continue

            sub_id, _ = self.parse_participant_session(participant_id)
            if not self.skip_participants_tsv:
                self._update_participants_tsv(sub_id)

            for chunk_key, chunk_data in chunks.items():
                out_dir, base_stem, run_index, block, order = self._build_base_entities(participant_id, chunk_key)

                for eyetracking_type in ['samples', 'fixations', 'saccades']:
                    if skip_samples and eyetracking_type == 'samples':
                        continue
                    df = chunk_data[eyetracking_type]
                    if df.empty:
                        continue

                    df = df.copy()
                    df['source_chunk_key'] = chunk_key
                    df['run_index'] = run_index
                    df['block'] = block
                    df['order'] = order

                    if eyetracking_type == 'samples':
                        df['AssignedCategory'] = ''
                        df['Category'] = ''
                        df['Designation'] = ''
                        for idx, row in df.iterrows():
                            pixel = row.get('MatchingPixel', -1)
                            if pixel >= 0:
                                cat = self.category_mapping.get(int(pixel), ('', '', ''))
                                df.at[idx, 'AssignedCategory'] = cat[0]
                                df.at[idx, 'Category'] = cat[1]
                                df.at[idx, 'Designation'] = cat[2]

                        sfreq = self._infer_sampling_frequency(df)
                        for rec_label, eye_prefix, recorded_eye in [('eye1', 'L', 'left'), ('eye2', 'R', 'right')]:
                            tsv_path = os.path.join(out_dir, f'{base_stem}_recording-{rec_label}_physio.tsv.gz')
                            out_df = self._sample_eye_dataframe(df, eye_prefix)
                            try:
                                self._write_tsv_gz(out_df, tsv_path)
                                json_path = tsv_path[:-7] + '.json'
                                sidecar = {
                                    'SamplingFrequency': sfreq,
                                    'StartTime': 0.0,
                                    'Columns': out_df.columns.tolist(),
                                    'PhysioType': 'eyetrack',
                                    'RecordedEye': recorded_eye,
                                    'SampleCoordinateSystem': 'gaze-on-screen',
                                    'SampleCoordinateUnits': 'pixel',
                                    'Manufacturer': 'SR Research',
                                    'ManufacturersModelName': 'EyeLink 1000',
                                    'EyeTrackingMethod': 'pupil',
                                    'Description': 'Eye-tracking samples with semantic category assignments.',
                                    'timestamp': {
                                        'Description': 'Eye-tracker timestamp for the sampled eye.',
                                        'Units': 'ms',
                                    },
                                    'x_coordinate': {
                                        'Description': 'Horizontal gaze position on screen.',
                                        'Units': 'pixel',
                                    },
                                    'y_coordinate': {
                                        'Description': 'Vertical gaze position on screen.',
                                        'Units': 'pixel',
                                    },
                                    'pupil_size': {
                                        'Description': 'Pupil size reported by the eye tracker.',
                                        'Units': 'arbitrary',
                                    },
                                    'SourceChunkKey': chunk_key,
                                }
                                if block is not None:
                                    sidecar['Block'] = block
                                if order is not None:
                                    sidecar['Order'] = order
                                self._write_json(json_path, sidecar)
                            except Exception as e:
                                self.logger.error('Failed to save %s: %s', tsv_path, e)
                        continue

                    df['AssignedCategory_Start'] = ''
                    df['AssignedCategory_End'] = ''
                    df['Category_Start'] = ''
                    df['Category_End'] = ''
                    df['Designation_Start'] = ''
                    df['Designation_End'] = ''
                    for idx, row in df.iterrows():
                        pixel_start = row.get('MatchingPixel_Start', -1)
                        pixel_end = row.get('MatchingPixel_End', -1)
                        if pixel_start >= 0:
                            cat_start = self.category_mapping.get(int(pixel_start), ('', '', ''))
                            df.at[idx, 'AssignedCategory_Start'] = cat_start[0]
                            df.at[idx, 'Category_Start'] = cat_start[1]
                            df.at[idx, 'Designation_Start'] = cat_start[2]
                        if pixel_end >= 0:
                            cat_end = self.category_mapping.get(int(pixel_end), ('', '', ''))
                            df.at[idx, 'AssignedCategory_End'] = cat_end[0]
                            df.at[idx, 'Category_End'] = cat_end[1]
                            df.at[idx, 'Designation_End'] = cat_end[2]

                    if 'tStart' in df.columns:
                        df['onset'] = pd.to_numeric(df['tStart'], errors='coerce') / 1000.0
                    if 'duration' in df.columns:
                        df['duration'] = pd.to_numeric(df['duration'], errors='coerce') / 1000.0

                    tsv_path = os.path.join(out_dir, f'{base_stem}_recording-{eyetracking_type}_physioevents.tsv.gz')
                    try:
                        out_df = self._physioevents_dataframe(df)
                        self._write_tsv_gz(out_df, tsv_path)
                        json_path = tsv_path[:-7] + '.json'
                        sidecar = {
                            'StartTime': 0.0,
                            'Columns': out_df.columns.tolist(),
                            'OnsetSource': 'timestamp',
                            'Manufacturer': 'SR Research',
                            'ManufacturersModelName': 'EyeLink 1000',
                            'EyeTrackingMethod': 'pupil',
                            'Description': f'{eyetracking_type} events with semantic category assignments.',
                            'SourceChunkKey': chunk_key,
                        }
                        if block is not None:
                            sidecar['Block'] = block
                        if order is not None:
                            sidecar['Order'] = order
                        self._write_json(json_path, sidecar)
                    except Exception as e:
                        self.logger.error('Failed to save %s: %s', tsv_path, e)

        self.logger.info('StreamAnalysis04 complete')
        return True
