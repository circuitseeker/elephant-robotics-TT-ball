# Elephant Robotics MyAGV - TT Ball Tracker

Autonomous ping pong ball detection and retrieval system using MyAGV 2023 (Raspberry Pi 4) with remote YOLO inference on RTX 5090.

## Architecture

```
[Logitech C270] → [Raspberry Pi 4] --TCP/WiFi-→ [RTX 5090 YOLO11s] → detections
                        ↓
                  [RPLidar A1] → obstacle avoidance
                        ↓
                  [ROS /cmd_vel] → motor control
                        ↓
                  [Web Dashboard :8080] → manual controls + live video
```

## Hardware
- **Robot**: Elephant Robotics MyAGV 2023 (Raspberry Pi 4, 4GB)
- **Camera**: Logitech C270 HD Webcam
- **LiDAR**: Slamtec RPLidar A1 (CP2102 USB adapter)
- **Inference**: NVIDIA RTX 5090 (dual GPU) running YOLO11s
- **Optional**: Google Coral USB Accelerator (Edge TPU)

## Files

### Robot (Raspberry Pi) - `~/Desktop/19-06TT/`
| File | Description |
|------|-------------|
| `raspi_streamer.py` | Main robot controller - camera capture, inference client, chase/explore logic, web dashboard, LiDAR obstacle avoidance |

### Inference Server (RTX 5090) - `~/Desktop/19-06TT/`
| File | Description |
|------|-------------|
| `inference_server.py` | TCP socket server running custom YOLO11s model |

### Training Scripts (RTX 5090)
| File | Description |
|------|-------------|
| `merge_and_train.py` | Merge Roboflow datasets + train YOLO11s |
| `auto_label_and_train.py` | Auto-label new images + retrain |
| `train_coral.py` | Train YOLO11n + MobileNet for Coral Edge TPU |
| `train_mobilenet_gpu1.py` | MobileNet v2 training for Edge TPU |
| `download_and_train.py` | Download dataset + train |
| `capture_dataset.py` | Capture frames for dataset |

## Model
- **Architecture**: YOLO11s (custom trained)
- **Dataset**: Merged Roboflow ping pong ball datasets (~3300 train, ~490 val)
- **Performance**: mAP50: 94.7%, ~7 FPS over WiFi
- **Input**: 640x640, conf threshold 0.25

## Quick Start

### 1. Start Inference Server (RTX 5090)
```bash
ssh helium5090@<5090-IP>
cd ~/Desktop/19-06TT
python3 -u inference_server.py
```

### 2. Start Robot (Raspberry Pi)
```bash
ssh er@<raspi-IP>
# Terminal 1: Start ROS
source /opt/ros/noetic/setup.bash
source ~/myagv_ros/devel/setup.bash
roslaunch myagv_odometry myagv_active.launch

# Terminal 2: Start streamer
source /opt/ros/noetic/setup.bash
source ~/myagv_ros/devel/setup.bash
python3 -u ~/Desktop/19-06TT/raspi_streamer.py
```

### 3. Open Dashboard
Navigate to `http://<raspi-IP>:8080`

## Dashboard Features
- Live camera feed with detection overlays
- Mode indicator (IDLE / CHASE / EXPLORE / CONFIRM / MANUAL)
- AUTO toggle for autonomous ball chasing + exploration
- Hold-to-move drive controls (forward, backward, left, right)
- Crab (strafe) controls
- Motor speed slider (10-100%)
- Server restart / stop motors

## Autonomous Behavior
1. **Explore**: Drive forward, rotate 60°, pause to look for balls (LiDAR obstacle avoidance)
2. **Confirm**: Ball detected → sliding window confirmation (5/7 frames, conf ≥ 0.4)
3. **Chase**: Proportional steering toward confirmed ball with obstacle avoidance
4. **Search**: Ball lost → return to explore mode

## Configuration
Edit `raspi_streamer.py` top section:
- `INFERENCE_MODE`: `"remote"` (5090) or `"coral"` (Edge TPU)
- `INFERENCE_HOST`: 5090 LAN IP
- `CAMERA_DEVICE`: `/dev/video1` (Logitech) or `/dev/video0` (Pi camera)
- Speed/obstacle thresholds: `BASE_LINEAR`, `OBS_STOP`, `OBS_SLOW`, etc.

## Requirements

### Raspberry Pi
- ROS Noetic
- OpenCV, rplidar library
- `myagv_odometry` package (for motor control via `/cmd_vel`)

### RTX 5090
- PyTorch + ultralytics
- Custom trained model: `best_ttball.pt`
