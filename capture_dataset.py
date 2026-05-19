"""
Capture frames from the Logitech C270 for training data.
Saves frames every 0.5s. Move the ball around during capture.
Press Ctrl+C on the raspi to stop.

Run on: er@100.82.170.22
"""

import cv2
import os
import time

CAMERA_DEVICE = "/dev/video1"
SAVE_DIR = os.path.expanduser("~/Desktop/19-06TT/raw_frames")
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

print(f"Camera opened. Saving frames to {SAVE_DIR}")
print("Move the ball around! Capture different positions, distances, lighting.")
print("Press Ctrl+C to stop.")

count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        fname = os.path.join(SAVE_DIR, f"frame_{count:04d}.jpg")
        cv2.imwrite(fname, frame)
        count += 1
        print(f"  Captured {count} frames", end='\r')
        time.sleep(0.5)
except KeyboardInterrupt:
    pass

cap.release()
print(f"\nDone. Captured {count} frames in {SAVE_DIR}")
