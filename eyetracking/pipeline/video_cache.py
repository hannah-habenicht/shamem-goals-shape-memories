# Lightweight video metadata cache 
import cv2

class VideoMetadataCache:
    def __init__(self):
        self._cache = {}

    def get(self, video_path):
        """Return dict with keys: frame_rate, total_frames, frame_timestamps.
        Caches results on first read.
        """
        if video_path in self._cache:
            return self._cache[video_path]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_rate <= 0:
            raise ValueError(f"Could not read frame rate from video: {video_path}")
        if total_frames <= 0:
            raise ValueError(f"Could not read frame count from video (got {total_frames}): {video_path}")

        frame_duration_ms = 1000.0 / frame_rate
        frame_timestamps = [i * frame_duration_ms for i in range(total_frames)]

        info = {
            'frame_rate': frame_rate,
            'total_frames': total_frames,
            'frame_timestamps': frame_timestamps,
        }
        self._cache[video_path] = info
        return info

    def initialize_from_dict(self, data):
        """Pre-populate the cache with externally computed metadata.

        Call this in worker processes with data that was computed once in the
        main process to avoid multiple workers opening the same video files
        concurrently (which can trigger H264 decoder errors in OpenCV).
        """
        self._cache.update(data)
