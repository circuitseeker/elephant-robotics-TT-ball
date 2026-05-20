"""
Train YOLO11n + SSD MobileNet for Coral Edge TPU deployment.
Run on: helium5090@100.97.153.40 (in ~/Desktop/19-06TT/)

Approach 1: YOLO11n (nano) - smaller, quantizes better
Approach 2: SSD MobileNet v2 - native Edge TPU support
"""

import os
import subprocess
import sys

WORK = os.path.expanduser("~/Desktop/19-06TT")
MERGED = os.path.join(WORK, "merged_dataset")
YAML = os.path.join(MERGED, "data.yaml")


def train_yolo_nano():
    """Train YOLO11n and export for Edge TPU with calibration data."""
    from ultralytics import YOLO

    print("=" * 60)
    print("APPROACH 1: YOLO11n for Edge TPU")
    print("=" * 60)

    model = YOLO("yolo11n.pt")

    model.train(
        data=YAML,
        epochs=150,
        imgsz=320,
        batch=128,
        name="ttball_nano_coral",
        project=os.path.join(WORK, "runs"),
        patience=30,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        close_mosaic=15,
        copy_paste=0.3,
        scale=0.5,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        fliplr=0.5,
        workers=8,
        exist_ok=True,
        verbose=True,
        save=True,
        val=True,
        plots=True,
        seed=42,
        device=0,
        int8=True,
    )

    best = os.path.join(WORK, "runs", "ttball_nano_coral", "weights", "best.pt")
    print(f"\nBest weights: {best}")

    print("\nExporting to Edge TPU with calibration data...")
    best_model = YOLO(best)
    best_model.export(
        format="edgetpu",
        imgsz=320,
        int8=True,
        data=YAML,
    )

    edgetpu_model = best.replace(".pt", "_full_integer_quant_edgetpu.tflite")
    saved_model_dir = best.replace(".pt", "_saved_model")
    candidate = os.path.join(saved_model_dir, "best_full_integer_quant_edgetpu.tflite")

    for path in [edgetpu_model, candidate]:
        if os.path.exists(path):
            final = os.path.join(WORK, "yolo11n_edgetpu.tflite")
            os.system(f"cp '{path}' '{final}'")
            print(f"Edge TPU model: {final}")
            return final

    print("Trying manual edgetpu_compiler...")
    tflite_dir = saved_model_dir
    for f in os.listdir(tflite_dir):
        if "integer_quant" in f and f.endswith(".tflite") and "edgetpu" not in f:
            src = os.path.join(tflite_dir, f)
            subprocess.run(["edgetpu_compiler", src, "-o", WORK], check=True)
            final = os.path.join(WORK, f.replace(".tflite", "_edgetpu.tflite"))
            if os.path.exists(final):
                print(f"Edge TPU model: {final}")
                return final

    print("WARNING: Could not find Edge TPU compiled model")
    return None


def train_mobilenet_ssd():
    """Fine-tune SSD MobileNet v2 for Edge TPU using TensorFlow."""
    print("\n" + "=" * 60)
    print("APPROACH 2: SSD MobileNet v2 for Edge TPU")
    print("=" * 60)

    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow not available, skipping MobileNet approach")
        return None

    import glob
    import cv2
    import numpy as np

    train_imgs = glob.glob(os.path.join(MERGED, "train", "images", "*"))
    val_imgs = glob.glob(os.path.join(MERGED, "valid", "images", "*"))
    print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")

    IMG_SIZE = 320

    def load_dataset(img_paths, img_size=IMG_SIZE):
        images = []
        boxes_list = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype(np.float32) / 255.0)

            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            bboxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = cx - bw / 2
                            y1 = cy - bh / 2
                            x2 = cx + bw / 2
                            y2 = cy + bh / 2
                            bboxes.append([y1, x1, y2, x2])
            if not bboxes:
                bboxes = [[0, 0, 0, 0]]
            boxes_list.append(bboxes[0])
        return np.array(images), np.array(boxes_list, dtype=np.float32)

    print("Loading training data...")
    train_x, train_boxes = load_dataset(train_imgs)
    val_x, val_boxes = load_dataset(val_imgs)
    print(f"Loaded: train={len(train_x)}, val={len(val_x)}")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    conf_out = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence")(x)
    box_out = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)

    model = tf.keras.Model(inputs=base.input, outputs=[conf_out, box_out])

    train_conf = (np.sum(np.abs(train_boxes), axis=1) > 0).astype(np.float32)
    val_conf = (np.sum(np.abs(val_boxes), axis=1) > 0).astype(np.float32)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
    )

    print("Training MobileNet SSD (frozen base)...")
    model.fit(
        train_x, {"confidence": train_conf, "bbox": train_boxes},
        validation_data=(val_x, {"confidence": val_conf, "bbox": val_boxes}),
        epochs=20,
        batch_size=32,
    )

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
    )

    print("Fine-tuning top layers...")
    model.fit(
        train_x, {"confidence": train_conf, "bbox": train_boxes},
        validation_data=(val_x, {"confidence": val_conf, "bbox": val_boxes}),
        epochs=30,
        batch_size=32,
    )

    saved_path = os.path.join(WORK, "mobilenet_ttball")
    model.save(saved_path)
    print(f"Model saved: {saved_path}")

    print("Converting to TFLite int8...")

    def representative_data_gen():
        for i in range(min(200, len(train_x))):
            yield [np.expand_dims(train_x[i], 0)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()

    tflite_path = os.path.join(WORK, "mobilenet_ttball_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model: {tflite_path}")

    print("Compiling for Edge TPU...")
    subprocess.run(["edgetpu_compiler", tflite_path, "-o", WORK], check=True)
    edgetpu_path = tflite_path.replace(".tflite", "_edgetpu.tflite")
    if os.path.exists(edgetpu_path):
        print(f"Edge TPU model: {edgetpu_path}")
        return edgetpu_path

    print("WARNING: Edge TPU compilation may have failed")
    return None


if __name__ == "__main__":
    yolo_result = train_yolo_nano()
    mobilenet_result = train_mobilenet_ssd()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if yolo_result:
        print(f"  YOLO11n Edge TPU: {yolo_result}")
    if mobilenet_result:
        print(f"  MobileNet Edge TPU: {mobilenet_result}")
