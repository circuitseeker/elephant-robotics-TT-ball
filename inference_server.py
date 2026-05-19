"""
TT Ball inference server — TCP socket, custom trained YOLO11s.
Run on: helium5090@100.97.153.40 (in ~/Desktop/19-06TT/)
"""

import socket
import struct
import json
import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "/home/helium5090/Desktop/19-06TT/best_ttball.pt"
IMGSZ = 640
CONF_THRESHOLD = 0.25
PORT = 5556

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
model(dummy, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
model(dummy, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
print("Model warmup done.")


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            return None
        buf += chunk
    return buf


def handle_client(conn, addr):
    print(f"Client connected: {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(None)
    inf_count = 0
    t_start = time.time()

    while True:
        header = recv_exact(conn, 4)
        if header is None:
            break

        size = struct.unpack('>I', header)[0]
        if size > 5_000_000 or size == 0:
            break

        jpg_data = recv_exact(conn, size)
        if jpg_data is None:
            break

        nparr = np.frombuffer(jpg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            resp = b'{"d":[]}\n'
            conn.sendall(struct.pack('>I', len(resp)) + resp)
            continue

        results = model(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, round(conf, 3)])

        resp = json.dumps({"d": detections}).encode() + b'\n'
        conn.sendall(struct.pack('>I', len(resp)) + resp)

        inf_count += 1
        if inf_count % 200 == 0:
            elapsed = time.time() - t_start
            print(f"  {inf_count} inferences, {inf_count/elapsed:.1f} fps")

    conn.close()
    elapsed = time.time() - t_start
    fps = inf_count / elapsed if elapsed > 0 else 0
    print(f"Client disconnected: {addr} ({inf_count} frames, {fps:.1f} fps)")


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', PORT))
    srv.listen(2)
    print(f"Listening on :{PORT} | conf>{CONF_THRESHOLD}")

    while True:
        conn, addr = srv.accept()
        handle_client(conn, addr)


if __name__ == '__main__':
    main()
