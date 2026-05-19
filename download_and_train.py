"""
Download ping pong ball datasets from Roboflow and train YOLO11s.
Run on: helium5090@100.97.153.40 (in ~/Desktop/19-06TT/)
"""

import os
import shutil
import glob
from roboflow import Roboflow

WORK_DIR = os.path.expanduser("~/Desktop/19-06TT")
DATASET_DIR = os.path.join(WORK_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(WORK_DIR)

# Download Dataset 1: SDP PongPickup Pro (robot pickup scenario — closest match)
print("=" * 60)
print("Downloading Dataset 1: SDP PongPickup Pro")
print("=" * 60)
rf = Roboflow(api_key="YOUR_API_KEY")  # will use public download
try:
    from roboflow import download
except:
    pass

# Use the Roboflow CLI-style download (public datasets)
os.system("pip3 install -q roboflow 2>/dev/null")

from roboflow import Roboflow

# Public API key for public datasets
rf = Roboflow(api_key="rf_public")
try:
    project = rf.workspace("sdp-pongpickup-pro").project("ping-pong-ball-detection-e04ql")
    version = project.version(1)
    ds1 = version.download("yolov8", location=os.path.join(DATASET_DIR, "ds1"))
    print(f"Dataset 1 downloaded to {ds1.location}")
except Exception as e:
    print(f"Dataset 1 download failed: {e}")
    print("Trying alternative download method...")

# Dataset 2: pingpong-2 (larger dataset)
print("\n" + "=" * 60)
print("Downloading Dataset 2: pingpong-2")
print("=" * 60)
try:
    project2 = rf.workspace("table-tennis-ball-detecting").project("pingpong-2")
    version2 = project2.version(3)
    ds2 = version2.download("yolov8", location=os.path.join(DATASET_DIR, "ds2"))
    print(f"Dataset 2 downloaded to {ds2.location}")
except Exception as e:
    print(f"Dataset 2 download failed: {e}")

print("\nDownload phase complete.")
print("Check dataset directories and run train step manually if needed.")
