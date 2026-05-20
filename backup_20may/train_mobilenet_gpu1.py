"""Train MobileNet v2 on GPU 1 and export for Coral Edge TPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tensorflow as tf
import numpy as np
import cv2
import glob
import subprocess

WORK = os.path.expanduser("~/Desktop/19-06TT")
MERGED = os.path.join(WORK, "merged_dataset")
IMG_SIZE = 320

print("GPUs:", tf.config.list_physical_devices("GPU"))
strategy = tf.distribute.MirroredStrategy()
print(f"Using {strategy.num_replicas_in_sync} GPUs")

train_imgs = glob.glob(os.path.join(MERGED, "train", "images", "*"))
val_imgs = glob.glob(os.path.join(MERGED, "valid", "images", "*"))

def load_dataset(img_paths):
    images, boxes_list = [], []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32) / 255.0)
        lbl_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        bboxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, bw, bh = map(float, parts[1:5])
                        bboxes.append([cy - bh / 2, cx - bw / 2, cy + bh / 2, cx + bw / 2])
        if not bboxes:
            bboxes = [[0, 0, 0, 0]]
        boxes_list.append(bboxes[0])
    return np.array(images), np.array(boxes_list, dtype=np.float32)

print("Loading data...")
train_x, train_boxes = load_dataset(train_imgs)
val_x, val_boxes = load_dataset(val_imgs)
train_conf = (np.sum(np.abs(train_boxes), axis=1) > 0).astype(np.float32)
val_conf = (np.sum(np.abs(val_boxes), axis=1) > 0).astype(np.float32)
print(f"Train: {len(train_x)}, Val: {len(val_x)}")

with strategy.scope():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(320, 320, 3), include_top=False, weights="imagenet"
    )
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    conf_out = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence")(x)
    box_out = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)
    model = tf.keras.Model(inputs=base.input, outputs=[conf_out, box_out])

    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
    )

print("Phase 1: frozen base (20 epochs)...")
model.fit(
    train_x, {"confidence": train_conf, "bbox": train_boxes},
    validation_data=(val_x, {"confidence": val_conf, "bbox": val_boxes}),
    epochs=20, batch_size=64, verbose=2,
)

with strategy.scope():
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
    )

print("Phase 2: fine-tuning (30 epochs)...")
model.fit(
    train_x, {"confidence": train_conf, "bbox": train_boxes},
    validation_data=(val_x, {"confidence": val_conf, "bbox": val_boxes}),
    epochs=30, batch_size=64, verbose=2,
)

saved_path = os.path.join(WORK, "mobilenet_ttball")
model.export(saved_path)
print(f"Saved: {saved_path}")

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
print(f"TFLite: {tflite_path}")

print("Compiling for Edge TPU...")
subprocess.run(["edgetpu_compiler", tflite_path, "-o", WORK], check=True)
print(f"Edge TPU model: {WORK}/mobilenet_ttball_int8_edgetpu.tflite")
print("ALL DONE")
