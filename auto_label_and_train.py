"""
Auto-label ping pong ball dataset using COCO-pretrained YOLO11 ('sports ball' class=32),
then fine-tune a new model specifically for TT ball detection.

Step 1: Capture frames from the raspi camera feed
Step 2: Auto-label using YOLO11s COCO model (sports ball = class 32)
Step 3: Train a fine-tuned model

Run on: helium5090@100.97.153.40 (in ~/Desktop/19-06TT/)
"""

import os
import sys
import cv2
import glob
import json
import shutil
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

WORK_DIR = os.path.expanduser("~/Desktop/19-06TT")
FRAMES_DIR = os.path.join(WORK_DIR, "captured_frames")
DATASET_DIR = os.path.join(WORK_DIR, "ttball_dataset")

# COCO class 32 = sports ball
SPORTS_BALL_CLASS = 32


def step1_capture_from_stream(url, num_frames=500, interval_ms=200):
    """Capture frames from the raspi MJPEG stream."""
    os.makedirs(FRAMES_DIR, exist_ok=True)

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"ERROR: Cannot open stream {url}")
        return 0

    count = 0
    print(f"Capturing {num_frames} frames from {url}...")
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            continue
        fname = os.path.join(FRAMES_DIR, f"frame_{count:04d}.jpg")
        cv2.imwrite(fname, frame)
        count += 1
        if count % 50 == 0:
            print(f"  Captured {count}/{num_frames}")
        cv2.waitKey(interval_ms)

    cap.release()
    print(f"Captured {count} frames")
    return count


def step2_auto_label(conf_threshold=0.15):
    """Use COCO YOLO11s to find 'sports ball' in each frame."""
    print("\n" + "=" * 60)
    print("Step 2: Auto-labeling with YOLO11s COCO model")
    print("=" * 60)

    model = YOLO("yolo11s.pt")

    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    print(f"Processing {len(frames)} frames...")

    labeled = 0
    empty = 0
    labels_dir = os.path.join(FRAMES_DIR, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    for i, fpath in enumerate(frames):
        frame = cv2.imread(fpath)
        h, w = frame.shape[:2]

        results = model(frame, imgsz=640, conf=conf_threshold, verbose=False)

        lines = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == SPORTS_BALL_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path = os.path.join(labels_dir, Path(fpath).stem + ".txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

        if lines:
            labeled += 1
        else:
            empty += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(frames)}, labeled: {labeled}, empty: {empty}")

    print(f"\nLabeling complete: {labeled} with detections, {empty} empty (negatives)")
    return labeled


def step3_prepare_dataset(train_ratio=0.8):
    """Organize into YOLO dataset structure."""
    print("\n" + "=" * 60)
    print("Step 3: Preparing dataset")
    print("=" * 60)

    for split in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        os.makedirs(os.path.join(DATASET_DIR, split), exist_ok=True)

    frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    random.shuffle(frames)

    split_idx = int(len(frames) * train_ratio)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    for split_name, split_frames in [('train', train_frames), ('val', val_frames)]:
        for fpath in split_frames:
            stem = Path(fpath).stem
            label_path = os.path.join(FRAMES_DIR, "labels", stem + ".txt")

            shutil.copy2(fpath, os.path.join(DATASET_DIR, split_name, "images", Path(fpath).name))
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(DATASET_DIR, split_name, "labels", stem + ".txt"))

    # Write data.yaml
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("nc: 1\n")
        f.write("names: ['ping_pong_ball']\n")

    print(f"Dataset prepared: {len(train_frames)} train, {len(val_frames)} val")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


def step4_train(yaml_path):
    """Fine-tune YOLO11s on the TT ball dataset."""
    print("\n" + "=" * 60)
    print("Step 4: Training YOLO11s")
    print("=" * 60)

    model = YOLO("yolo11s.pt")

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=-1,
        name="ttball_v1",
        project=os.path.join(WORK_DIR, "runs"),
        patience=25,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        close_mosaic=15,
        copy_paste=0.2,
        scale=0.5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        workers=8,
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=25,
        val=True,
        plots=True,
        seed=42,
        device=0,
    )

    best_path = os.path.join(WORK_DIR, "runs", "ttball_v1", "weights", "best.pt")
    print(f"\nTraining complete!")
    print(f"Best weights: {best_path}")
    return best_path


if __name__ == '__main__':
    # Step 1: Capture frames from raspi stream
    stream_url = "http://100.82.170.22:8080/stream"
    n = step1_capture_from_stream(stream_url, num_frames=400, interval_ms=250)

    if n == 0:
        print("No frames captured. Exiting.")
        sys.exit(1)

    # Step 2: Auto-label
    labeled = step2_auto_label(conf_threshold=0.10)

    # Step 3: Prepare dataset
    yaml_path = step3_prepare_dataset()

    # Step 4: Train
    best_path = step4_train(yaml_path)

    print(f"\n{'=' * 60}")
    print(f"DONE! Deploy with: {best_path}")
    print(f"{'=' * 60}")
