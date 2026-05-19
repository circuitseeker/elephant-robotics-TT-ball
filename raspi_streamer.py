"""
Raspi Camera Streamer + MyAGV Controller + RPLidar Explorer
- Ball detection via Coral Edge TPU (local) or RTX 5090 (remote)
- Autonomous ball-chasing with confirmation window
- LiDAR-based exploration when no ball found
- Obstacle avoidance during chase and explore
- Web dashboard with manual controls, crab, speed slider
Run on: er@100.82.170.22
Requires: roslaunch myagv_odometry myagv_active.launch (running)
"""

import cv2
import time
import json
import struct
import socket
import os
import sys
import threading
import numpy as np
import signal
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import rospy
from geometry_msgs.msg import Twist

# ---- INFERENCE MODE: "coral" or "remote" ----
INFERENCE_MODE = "coral"

if INFERENCE_MODE == "coral":
    import tflite_runtime.interpreter as tflite

# Remote (5090) settings
INFERENCE_HOST = "192.168.0.147"
INFERENCE_PORT = 5556

# Coral Edge TPU settings
MODEL_PATH = os.path.expanduser("~/Desktop/19-06TT/yolo11n_edgetpu.tflite")
IMGSZ = 320
CONF_THRESHOLD = 0.30
CAMERA_DEVICE = "/dev/video1"

SHM_FILE = "/dev/shm/ttball_frame.jpg"
STATS_FILE = "/dev/shm/ttball_stats.json"

FRAME_W = 640
FRAME_H = 480
FRAME_CX = FRAME_W // 2

# --- Shared state ---
cmd_pub = None
chase_enabled = False
manual_cmd = None
last_dets = []
stats = {"fps_capture": 0.0, "fps_inference": 0.0, "det_count": 0, "latency_ms": 0.0, "mode": "idle"}

# LiDAR data — updated by lidar thread
lidar_sectors = [10000.0] * 12  # 12 sectors of 30 deg each, min distance in mm
lidar_ok = False

# Speed settings
speed_mult = 1.0
BASE_LINEAR = 0.5
BASE_STRAFE = 0.4
BASE_ANGULAR = 1.0
SLOW_ANGULAR = 0.4

# Obstacle avoidance thresholds (mm)
OBS_STOP = 250       # emergency stop
OBS_SLOW = 500       # slow down
OBS_TURN = 600       # start turning away

# Explore settings
EXPLORE_SPEED = 0.25
EXPLORE_ANGULAR = 0.4


def publish_twist(lx=0.0, ly=0.0, az=0.0):
    if cmd_pub is None:
        return
    t = Twist()
    t.linear.x = lx
    t.linear.y = ly
    t.angular.z = az
    cmd_pub.publish(t)


def stop_robot():
    publish_twist(0, 0, 0)


def get_cmd_twist(cmd):
    s = max(0.1, speed_mult)
    m = {
        "forward":    (BASE_LINEAR * s, 0, 0),
        "backward":   (-BASE_LINEAR * s, 0, 0),
        "left":       (0, 0, BASE_ANGULAR * s),
        "right":      (0, 0, -BASE_ANGULAR * s),
        "crab_left":  (0, BASE_STRAFE * s, 0),
        "crab_right": (0, -BASE_STRAFE * s, 0),
        "stop":       (0, 0, 0),
    }
    return m.get(cmd, (0, 0, 0))


def get_front_clearance():
    """Get min distance in front 90 degrees (sectors 10,11,0,1 = -60 to +60 deg)."""
    front_sectors = [11, 0, 1]
    return min(lidar_sectors[i] for i in front_sectors)


def get_best_direction():
    """Find the most open direction to explore. Returns angular velocity to turn toward it."""
    best_sector = max(range(12), key=lambda i: lidar_sectors[i])
    best_angle = best_sector * 30  # 0=front-right, going clockwise

    # Convert sector to turn direction
    # Sectors: 0=0°(front-right), 1=30°, 2=60°, ... 6=180°(back), ... 11=330°(front-left)
    if best_sector <= 6:
        return -EXPLORE_ANGULAR  # turn right (clockwise)
    else:
        return EXPLORE_ANGULAR   # turn left (counter-clockwise)


def is_path_clear_toward(norm_offset):
    """Check if path toward ball (given by normalized camera offset) is clear."""
    # Map camera offset to lidar sector
    # Camera center = sector 0 (front), left = sector 11, right = sector 1
    if abs(norm_offset) < 0.2:
        sectors = [11, 0, 1]
    elif norm_offset < 0:
        sectors = [10, 11, 0]  # looking left
    else:
        sectors = [0, 1, 2]    # looking right
    return min(lidar_sectors[i] for i in sectors) > OBS_STOP


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            return None
        buf += chunk
    return buf


def load_edgetpu_model():
    """Load Edge TPU model and return interpreter with input/output details."""
    interp = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()
    print(f"Edge TPU model loaded: input={inp['shape']}, dtype={inp['dtype']}")
    for i, o in enumerate(out):
        print(f"  output[{i}]: shape={o['shape']}, dtype={o['dtype']}")
    return interp, inp, out


def run_edgetpu_inference(interp, inp_detail, out_details, frame):
    """Run inference on a frame, return detections as [[x1,y1,x2,y2,conf], ...]."""
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (IMGSZ, IMGSZ))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if inp_detail['dtype'] == np.uint8:
        inp_data = img.astype(np.uint8)
    elif inp_detail['dtype'] == np.int8:
        inp_data = (img.astype(np.int32) - 128).astype(np.int8)
    else:
        inp_data = (img.astype(np.float32) / 255.0)

    interp.set_tensor(inp_detail['index'], np.expand_dims(inp_data, 0))
    interp.invoke()

    raw = interp.get_tensor(out_details[0]['index'])
    out_q = out_details[0]
    if out_q['dtype'] != np.float32:
        scale, zp = out_q['quantization']
        raw = (raw.astype(np.float32) - zp) * scale

    raw = np.squeeze(raw)
    if raw.shape[0] == 5 and raw.shape[1] > 5:
        raw = raw.T

    detections = []
    for det in raw:
        cx, cy, bw, bh, conf = det[0], det[1], det[2], det[3], det[4]
        if conf < CONF_THRESHOLD:
            continue
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        detections.append([x1, y1, x2, y2, round(float(conf), 3)])

    if len(detections) > 1:
        detections.sort(key=lambda d: d[4], reverse=True)
        keep = []
        for d in detections:
            overlap = False
            for k in keep:
                ix1 = max(d[0], k[0]); iy1 = max(d[1], k[1])
                ix2 = min(d[2], k[2]); iy2 = min(d[3], k[3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                a1 = (d[2]-d[0])*(d[3]-d[1])
                a2 = (k[2]-k[0])*(k[3]-k[1])
                iou = inter / (a1+a2-inter+1e-6)
                if iou > 0.5:
                    overlap = True; break
            if not overlap:
                keep.append(d)
        detections = keep

    return detections


# ======================== LIDAR THREAD ========================

_lidar_instance = None

def run_lidar():
    """Continuously read RPLidar scans and update sector distances."""
    global lidar_sectors, lidar_ok, _lidar_instance

    try:
        from rplidar import RPLidar
    except ImportError:
        print("RPLidar library not installed, exploration disabled")
        return

    while True:
        try:
            lidar = RPLidar('/dev/ttyUSB1', baudrate=256000, timeout=3)
            _lidar_instance = lidar
            info = lidar.get_info()
            print(f"LiDAR connected: model={info['model']}, fw={info['firmware']}")
            health = lidar.get_health()
            print(f"LiDAR health: {health[0]}")
            lidar_ok = True

            for scan in lidar.iter_scans():
                sectors = [10000.0] * 12
                for _, angle, dist in scan:
                    if dist < 10:
                        continue
                    sec = int(angle / 30) % 12
                    if dist < sectors[sec]:
                        sectors[sec] = dist
                lidar_sectors = sectors

                if rospy.is_shutdown():
                    break

        except Exception as e:
            lidar_ok = False
            print(f"LiDAR error: {e}")
            try:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
            except:
                pass
            _lidar_instance = None
            time.sleep(3)


def cleanup_and_exit(signum=None, frame=None):
    """Stop LiDAR motor, stop robot, then exit."""
    global _lidar_instance
    print(f"\nShutting down (signal {signum})...")
    try:
        cmd_pub.publish(Twist())
    except:
        pass
    if _lidar_instance:
        try:
            _lidar_instance.stop()
            _lidar_instance.stop_motor()
            _lidar_instance.disconnect()
            print("LiDAR motor stopped")
        except:
            pass
        _lidar_instance = None
    os._exit(0)


# ======================== CAPTURE THREAD ========================

def open_camera():
    """Open camera with retry."""
    while not rospy.is_shutdown():
        cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"Camera: {int(cap.get(3))}x{int(cap.get(4))}")
            return cap
        print(f"Waiting for {CAMERA_DEVICE}...")
        cap.release()
        time.sleep(3)
    return None


def run_capture():
    """Main capture + inference loop. Supports both Coral and remote 5090."""
    global last_dets, stats

    cap = open_camera()
    if cap is None:
        return

    # Setup inference backend
    interp = inp_detail = out_details = None
    sock = None

    if INFERENCE_MODE == "coral":
        print("Loading Edge TPU model...")
        interp, inp_detail, out_details = load_edgetpu_model()
        print("Edge TPU ready")
    else:
        print(f"Using remote inference: {INFERENCE_HOST}:{INFERENCE_PORT}")

    enc = [cv2.IMWRITE_JPEG_QUALITY, 30]
    out_enc = [cv2.IMWRITE_JPEG_QUALITY, 70]
    fc = 0
    ic = 0
    tw = time.time()
    fps_i = 0.0
    fail_count = 0

    while not rospy.is_shutdown():
        # Remote mode: connect if needed
        if INFERENCE_MODE == "remote" and sock is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((INFERENCE_HOST, INFERENCE_PORT))
                print(f"Connected to {INFERENCE_HOST}:{INFERENCE_PORT}")
            except Exception as e:
                print(f"Connect failed: {e}")
                sock = None
                time.sleep(2)
                continue

        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 30:
                print("Camera lost, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = open_camera()
                if cap is None:
                    return
                fail_count = 0
            continue
        fail_count = 0

        fc += 1
        t0 = time.time()

        try:
            if INFERENCE_MODE == "coral":
                dets = run_edgetpu_inference(interp, inp_detail, out_details, frame)
            else:
                _, buf = cv2.imencode('.jpg', frame, enc)
                jpg = buf.tobytes()
                sock.sendall(struct.pack('>I', len(jpg)) + jpg)
                hdr = recv_exact(sock, 4)
                if hdr is None:
                    raise ConnectionError()
                rsize = struct.unpack('>I', hdr)[0]
                rdata = recv_exact(sock, rsize)
                if rdata is None:
                    raise ConnectionError()
                dets = json.loads(rdata).get("d", [])

            lat = (time.time() - t0) * 1000
            ic += 1
            last_dets = dets

            for d in dets:
                x1, y1, x2, y2, conf = d[:5]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.line(frame, (FRAME_CX - 20, FRAME_H // 2), (FRAME_CX + 20, FRAME_H // 2), (0, 0, 255), 1)
            cv2.line(frame, (FRAME_CX, FRAME_H // 2 - 20), (FRAME_CX, FRAME_H // 2 + 20), (0, 0, 255), 1)

            mode = stats.get("mode", "idle")
            mode_colors = {"idle": (128, 128, 128), "chase": (0, 255, 0), "explore": (255, 165, 0),
                           "confirm": (0, 255, 255), "manual": (255, 255, 0)}
            mc = mode_colors.get(mode, (255, 255, 255))
            cv2.putText(frame, mode.upper(), (FRAME_W - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc, 2)

            tn = time.time()
            el = tn - tw
            if el >= 1.0:
                fps_i = ic / el
                stats.update({
                    "fps_capture": round(fc / el, 1),
                    "fps_inference": round(fps_i, 1),
                })
                fc = 0
                ic = 0
                tw = tn

            stats["det_count"] = len(dets)
            stats["latency_ms"] = round(lat, 1)
            stats["chase"] = chase_enabled

            try:
                with open(STATS_FILE, 'w') as f:
                    json.dump(stats, f)
            except:
                pass

            cv2.putText(frame, f"Balls:{len(dets)} FPS:{fps_i:.0f} Lat:{lat:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            _, out = cv2.imencode('.jpg', frame, out_enc)
            try:
                with open(SHM_FILE + ".tmp", 'wb') as f:
                    f.write(out.tobytes())
                os.replace(SHM_FILE + ".tmp", SHM_FILE)
            except:
                pass

        except Exception as e:
            print(f"Inference error: {e}")
            if INFERENCE_MODE == "remote":
                try:
                    sock.close()
                except:
                    pass
                sock = None


# ======================== CONTROL THREAD ========================

def run_control():
    """Control loop with 3 modes:
    1. Chase — ball confirmed, drive toward it with obstacle avoidance
    2. Explore — no ball, use LiDAR to roam and search
    3. Manual — dashboard button commands
    """
    global manual_cmd, chase_enabled, speed_mult

    DEAD_ZONE_X = 30
    CLOSE_AREA = 0.15
    MAX_ANGULAR = 0.6
    CONF_THRESHOLD = 0.4
    CONFIRM_FRAMES = 5
    WINDOW_SIZE = 7
    rate = rospy.Rate(10)

    det_history = []
    confirmed = False

    # Explore state
    explore_state = "drive"     # drive, turn, pause_look
    explore_timer = time.time()
    explore_turn_dir = 1.0
    DRIVE_DURATION = 2.0        # drive forward for 2s
    TURN_DURATION = 1.5         # turn for ~60 deg
    LOOK_DURATION = 1.0         # pause and look for ball

    while not rospy.is_shutdown():
        # --- Manual commands always take priority ---
        cmd = manual_cmd
        if cmd:
            manual_cmd = None
            if cmd == "chase_on":
                chase_enabled = True
                det_history.clear()
                confirmed = False
                explore_state = "pause_look"
                explore_timer = time.time()
                stats["mode"] = "explore"
                print("Chase ON — starting explore")
            elif cmd == "chase_off":
                chase_enabled = False
                stop_robot()
                det_history.clear()
                confirmed = False
                stats["mode"] = "idle"
                print("Chase OFF")
            elif cmd.startswith("speed:"):
                speed_mult = float(cmd.split(":")[1])
                print(f"Speed: {speed_mult:.0%}")
            elif cmd in ("forward", "backward", "left", "right", "crab_left", "crab_right", "stop"):
                lx, ly, az = get_cmd_twist(cmd)
                publish_twist(lx, ly, az)
                stats["mode"] = "manual"
            rate.sleep()
            continue

        if not chase_enabled:
            rate.sleep()
            continue

        # --- Check for ball detections ---
        dets = last_dets
        good_dets = [d for d in dets if len(d) > 4 and d[4] >= CONF_THRESHOLD]

        # Always track detection history
        det_history.append(len(good_dets) > 0)
        if len(det_history) > WINDOW_SIZE:
            det_history.pop(0)
        hits = sum(det_history)

        front = get_front_clearance()

        # ===== OBSTACLE EMERGENCY STOP =====
        if front < OBS_STOP and confirmed:
            stop_robot()
            stats["mode"] = "obstacle"
            rate.sleep()
            continue

        # ===== CONFIRMED: CHASE MODE =====
        if confirmed:
            if hits < 2:
                confirmed = False
                stop_robot()
                det_history.clear()
                explore_state = "pause_look"
                explore_timer = time.time()
                stats["mode"] = "explore"
                print("Ball lost → exploring")
                rate.sleep()
                continue

            if not good_dets:
                stop_robot()
                rate.sleep()
                continue

            stats["mode"] = "chase"
            best = max(good_dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
            x1, y1, x2, y2 = best[:4]
            bcx = (x1 + x2) // 2
            ball_area = ((x2 - x1) * (y2 - y1)) / (FRAME_W * FRAME_H)

            if ball_area >= CLOSE_AREA:
                stop_robot()
                print("Ball reached! Pausing 3s then exploring...")
                time.sleep(3)
                det_history.clear()
                confirmed = False
                explore_state = "drive"
                explore_timer = time.time()
                stats["mode"] = "explore"
                rate.sleep()
                continue

            offset_x = bcx - FRAME_CX
            norm_offset = offset_x / (FRAME_W / 2)
            s = max(0.1, speed_mult)
            az = -norm_offset * MAX_ANGULAR * s

            if abs(offset_x) < DEAD_ZONE_X:
                az = 0.0

            # Slow down if obstacle ahead during chase
            speed_scale = 1.0
            if front < OBS_SLOW:
                speed_scale = 0.3
            elif front < OBS_TURN:
                speed_scale = 0.6

            if not is_path_clear_toward(norm_offset):
                speed_scale = 0.2

            fwd = BASE_LINEAR * s * max(0.3, 1.0 - abs(norm_offset)) * speed_scale
            publish_twist(fwd, 0, az)
            rate.sleep()
            continue

        # ===== NOT CONFIRMED: check if we should confirm =====
        if hits >= CONFIRM_FRAMES:
            confirmed = True
            print(f"Ball confirmed ({hits}/{WINDOW_SIZE}) → chasing!")
            stats["mode"] = "chase"
            rate.sleep()
            continue

        # If we see some detections, pause to confirm
        if hits > 0 and explore_state != "pause_look":
            explore_state = "pause_look"
            explore_timer = time.time()
            stats["mode"] = "confirm"

        # ===== EXPLORE MODE — roam with LiDAR =====
        now = time.time()
        elapsed = now - explore_timer

        if explore_state == "pause_look":
            # Stand still, let camera check for ball
            stop_robot()
            stats["mode"] = "confirm" if hits > 0 else "explore"
            if elapsed >= LOOK_DURATION:
                if hits == 0:
                    # Nothing found, pick direction and drive
                    explore_state = "turn"
                    explore_timer = now
                    # Turn toward most open LiDAR sector
                    explore_turn_dir = get_best_direction()

        elif explore_state == "turn":
            stats["mode"] = "explore"
            publish_twist(0, 0, explore_turn_dir)
            if elapsed >= TURN_DURATION:
                explore_state = "drive"
                explore_timer = now
                stop_robot()

        elif explore_state == "drive":
            stats["mode"] = "explore"
            if front < OBS_STOP:
                # Too close — stop and turn
                stop_robot()
                explore_state = "turn"
                explore_timer = now
                explore_turn_dir = get_best_direction()
            elif front < OBS_TURN:
                # Getting close — slow down and start turning
                az = get_best_direction() * 0.5
                publish_twist(EXPLORE_SPEED * 0.4, 0, az)
            else:
                # Clear path — drive forward
                publish_twist(EXPLORE_SPEED, 0, 0)

            if elapsed >= DRIVE_DURATION:
                explore_state = "pause_look"
                explore_timer = now
                stop_robot()
                det_history.clear()

        rate.sleep()


# ======================== DASHBOARD ========================

DASHBOARD_HTML = b"""<!DOCTYPE html><html><head><title>MyAGV Ball Tracker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0a;color:#eee;font-family:'Segoe UI',monospace;padding:10px}
h1{color:#0f0;text-align:center;font-size:1.4em;margin:8px 0}
.top-bar{text-align:center;color:#aaa;font-size:0.85em;margin-bottom:8px}
#stats{color:#ff0;font-size:14px;text-align:center;margin:6px 0}
#mode-bar{text-align:center;margin:4px 0;font-size:16px;font-weight:bold}
.main{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
.video-box{flex:1;min-width:320px;max-width:660px}
.video-box img{width:100%;border:2px solid #0f0;border-radius:4px}
.ctrl-panel{flex:0 0 240px;display:flex;flex-direction:column;gap:8px}
.section{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:10px}
.section h3{color:#0f0;font-size:0.9em;margin-bottom:8px;border-bottom:1px solid #333;padding-bottom:4px}
.btn{padding:10px 8px;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold;
     text-align:center;transition:all 0.15s;user-select:none;font-family:monospace;-webkit-touch-callout:none}
.btn:active{transform:scale(0.95)}
.btn-go{background:#1a3a1a;color:#0f0}.btn-go:hover{background:#2a5a2a}
.btn-go.held{background:#2a6a2a;border-color:#0f0}
.btn-stop{background:#3a1a1a;color:#f44}.btn-stop:hover{background:#5a2a2a}
.btn-chase{background:#1a2a3a;color:#4af}.btn-chase:hover{background:#2a4a6a}
.btn-chase.active{background:#0a4a0a;color:#0f0;border-color:#0f0}
.dpad{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}
.dpad .btn{padding:14px 4px;font-size:18px}
.spacer{visibility:hidden}
.slider-wrap{display:flex;align-items:center;gap:8px}
.slider-wrap input[type=range]{flex:1;accent-color:#0f0}
#speedVal{color:#0f0;font-size:16px;min-width:40px;text-align:right}
</style></head><body>
<h1>MyAGV Ball Tracker</h1>
<div class="top-bar">Logitech C270 + YOLO11s + RPLidar | mAP50: 94.7%</div>
<div id="stats">Loading...</div>
<div id="mode-bar" style="color:#888">MODE: IDLE</div>
<div class="main">
  <div class="video-box"><img id="stream" src="/stream"/></div>
  <div class="ctrl-panel">
    <div class="section">
      <h3>MODE</h3>
      <div class="btn btn-chase" id="chaseBtn" onclick="toggleChase()">AUTO: OFF</div>
    </div>
    <div class="section">
      <h3>MOTOR SPEED</h3>
      <div class="slider-wrap">
        <span style="color:#888">Slow</span>
        <input type="range" id="speedSlider" min="10" max="100" value="100" oninput="setSpeed(this.value)">
        <span id="speedVal">100%</span>
      </div>
    </div>
    <div class="section">
      <h3>DRIVE (hold to move)</h3>
      <div class="dpad">
        <div class="spacer"></div>
        <div class="btn btn-go" data-cmd="forward">&#9650;</div>
        <div class="spacer"></div>
        <div class="btn btn-go" data-cmd="left">&#9664; L</div>
        <div class="btn btn-stop" data-cmd="stop">STOP</div>
        <div class="btn btn-go" data-cmd="right">R &#9654;</div>
        <div class="spacer"></div>
        <div class="btn btn-go" data-cmd="backward">&#9660;</div>
        <div class="spacer"></div>
      </div>
    </div>
    <div class="section">
      <h3>CRAB (strafe)</h3>
      <div style="display:flex;gap:4px">
        <div class="btn btn-go" data-cmd="crab_left" style="flex:1">&#8678; CRAB L</div>
        <div class="btn btn-go" data-cmd="crab_right" style="flex:1">CRAB R &#8680;</div>
      </div>
    </div>
    <div class="section">
      <h3>SERVER</h3>
      <div style="display:flex;gap:4px">
        <div class="btn btn-stop" onclick="if(confirm('Restart server?'))fetch('/restart')" style="flex:1">RESTART</div>
        <div class="btn btn-go" onclick="fetch('/cmd?c=stop')" style="flex:1">STOP MOTORS</div>
      </div>
    </div>
  </div>
</div>
<script>
let chaseOn=false,holdTimer=null,activeBtn=null,keysDown={};
function cmd(c){fetch('/cmd?c='+c)}
function setSpeed(v){document.getElementById('speedVal').textContent=v+'%';cmd('speed:'+(v/100))}
function toggleChase(){
  chaseOn=!chaseOn;
  cmd(chaseOn?'chase_on':'chase_off');
  updChase();
}
function updChase(){
  let b=document.getElementById('chaseBtn');
  b.textContent='AUTO: '+(chaseOn?'ON':'OFF');
  b.classList.toggle('active',chaseOn);
}
function startHold(btn,c){
  stopHold();
  activeBtn=btn;btn.classList.add('held');
  cmd(c);
  holdTimer=setInterval(()=>cmd(c),150);
}
function stopHold(){
  if(holdTimer){clearInterval(holdTimer);holdTimer=null}
  if(activeBtn){activeBtn.classList.remove('held');activeBtn=null}
  cmd('stop');
}
document.querySelectorAll('[data-cmd]').forEach(b=>{
  let c=b.dataset.cmd;
  b.addEventListener('mousedown',e=>{e.preventDefault();if(c==='stop'){cmd('stop')}else{startHold(b,c)}});
  b.addEventListener('touchstart',e=>{e.preventDefault();if(c==='stop'){cmd('stop')}else{startHold(b,c)}});
});
document.addEventListener('mouseup',stopHold);
document.addEventListener('touchend',stopHold);
document.addEventListener('touchcancel',stopHold);
const keyMap={ArrowUp:'forward',ArrowDown:'backward',ArrowLeft:'left',ArrowRight:'right',
  w:'forward',s:'backward',a:'left',d:'right',q:'crab_left',e:'crab_right',' ':'stop'};
document.addEventListener('keydown',e=>{
  let c=keyMap[e.key];if(!c||keysDown[e.key])return;e.preventDefault();
  keysDown[e.key]=true;cmd(c);
});
document.addEventListener('keyup',e=>{
  if(keysDown[e.key]){delete keysDown[e.key];if(Object.keys(keysDown).length===0)cmd('stop')}
});

const modeColors={idle:'#888',chase:'#0f0',explore:'#ffa500',confirm:'#0ff',manual:'#ff0',obstacle:'#f00'};
setInterval(()=>{fetch('/stats').then(r=>r.json()).then(d=>{
  document.getElementById('stats').innerText=
    'Capture: '+d.fps_capture+' fps | Inference: '+d.fps_inference+' fps | Balls: '+d.det_count+' | Latency: '+d.latency_ms+'ms';
  if(d.chase!==undefined){chaseOn=d.chase;updChase()}
  let m=d.mode||'idle';
  let mb=document.getElementById('mode-bar');
  mb.textContent='MODE: '+m.toUpperCase();
  mb.style.color=modeColors[m]||'#888';
}).catch(()=>{})},1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        global manual_cmd

        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                try:
                    with open(SHM_FILE, 'rb') as f:
                        jpg = f.read()
                    if jpg:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                    time.sleep(0.04)
                except:
                    break

        elif self.path == '/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                with open(STATS_FILE, 'r') as f:
                    self.wfile.write(f.read().encode())
            except:
                self.wfile.write(b'{}')

        elif self.path.startswith('/cmd'):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            c = q.get('c', [''])[0]
            if c:
                manual_cmd = c
                print(f"CMD: {c}")
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'ok')

        elif self.path == '/restart':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'restarting...')
            print("RESTART requested")
            stop_robot()
            rospy.signal_shutdown("restart")
            time.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    rospy.init_node('ball_tracker', anonymous=False, disable_signals=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)
    time.sleep(0.5)
    print("ROS node initialized, /cmd_vel publisher ready")

    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

    threads = [
        threading.Thread(target=run_capture, daemon=True),
        threading.Thread(target=run_control, daemon=True),
        threading.Thread(target=run_lidar, daemon=True),
    ]
    for t in threads:
        t.start()

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    print("Web server on :8080 (threaded)")
    server = ThreadedHTTPServer(('0.0.0.0', 8080), Handler)
    server.serve_forever()
