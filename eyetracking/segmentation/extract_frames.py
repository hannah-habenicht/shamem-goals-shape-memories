"""Extract every frame of each film to <FRAMES_DIR>/<film>/frame_XXXX.jpg."""
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config

for file_name in os.listdir(config.VIDEO_DIR):
    if not file_name.endswith(".mp4"):
        continue
    cap = cv2.VideoCapture(os.path.join(config.VIDEO_DIR, file_name))
    movie_folder = os.path.join(config.FRAMES_DIR, os.path.splitext(file_name)[0])
    os.makedirs(movie_folder, exist_ok=True)
    print(movie_folder)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(movie_folder, f"frame_{frame_count:04d}.jpg"), frame)
        frame_count += 1
    cap.release()
