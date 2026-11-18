"""Label every pixel of every film frame with an ADE20K class.

Model: ResNet101 encoder + UPerNet decoder, pre-trained on ADE20K (epoch 50), from
https://github.com/CSAILVision/semantic-segmentation-pytorch (commit 8f27c9b).
Each frame is resized so its short side is 300 px, normalised with the ImageNet mean and
SD by that repository's TestDataset, and the prediction is upsampled to the frame size.

Output: <MASKS_DIR>/<film>/<film>_frame_XXXX.csv, semicolon-separated with a header row and
an index column; values are 1-indexed ADE20K classes (1-150).

Usage:
    python segment_frames.py                                   # all films, all frames
    python segment_frames.py --films Charite --frames 100 1000 # a subset, e.g. to check a result
"""
import argparse
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import config

NUM_CLASS = 150
OPTIONS = SimpleNamespace(imgSizes=[300], imgMaxSize=1000, padding_constant=8)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--films", nargs="+", help="Film folder names (default: all in FRAMES_DIR)")
parser.add_argument("--frames", nargs="+", type=int, help="Frame numbers (default: all)")
args = parser.parse_args()

builder = ModelBuilder()
encoder = builder.build_encoder(arch="resnet101", fc_dim=2048,
                                weights=os.path.join(config.SEGMENTATION_WEIGHTS_DIR, "encoder_epoch_50.pth"))
decoder = builder.build_decoder(arch="upernet", fc_dim=2048, num_class=NUM_CLASS, use_softmax=True,
                                weights=os.path.join(config.SEGMENTATION_WEIGHTS_DIR, "decoder_epoch_50.pth"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SegmentationModule(encoder, decoder, torch.nn.NLLLoss(ignore_index=-1)).eval().to(device)

for film in args.films or sorted(os.listdir(config.FRAMES_DIR)):
    in_dir = os.path.join(config.FRAMES_DIR, film)
    out_dir = os.path.join(config.MASKS_DIR, film)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".jpg"))
    if args.frames:
        names = [f"frame_{i:04d}.jpg" for i in args.frames]
    for name in names:
        batch = TestDataset([{"fpath_img": os.path.join(in_dir, name)}], OPTIONS)[0]
        size = batch["img_ori"].shape[:2]
        with torch.no_grad():
            scores = model({"img_data": batch["img_data"][0].to(device)}, segSize=size)
        labels = scores.argmax(dim=1)[0].cpu().numpy() + 1
        pd.DataFrame(labels).to_csv(os.path.join(out_dir, f"{film}_{name[:-4]}.csv"), sep=";", index=True)
    print(f"{film}: {len(names)} frames")
