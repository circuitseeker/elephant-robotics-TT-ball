"""
Train SSD MobileNet v2 for Coral Edge TPU - Dual RTX 5090 GPUs.
Produces a quantized int8 TFLite model compiled for Edge TPU.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
import numpy as np
import cv2
import glob
import subprocess
import time

# Use both GPUs
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()
print(f"GPUs: {gpus}")
print(f"Replicas: {strategy.num_replicas_in_sync}")

MERGED = os.path.expanduser("~/Desktop/19-06TT/merged_dataset")
IMG_SIZE = 320
BATCH_SIZE = 128 * strategy.num_replicas_in_sync  # 256 total across 2 GPUs
NUM_CLASSES = 1

# ======================== DATA LOADING ========================

def load_dataset(split):
    img_dir = os.path.join(MERGED, split, "images")
    lbl_dir = os.path.join(MERGED, split, "labels")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))

    images = []
    boxes = []
    confs = []

    for ip in img_paths:
        img = cv2.imread(ip)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)

        lp = os.path.join(lbl_dir, os.path.splitext(os.path.basename(ip))[0] + ".txt")
        if os.path.exists(lp):
            with open(lp) as f:
                lines = f.readlines()
            if lines:
                # Take first object (single class)
                parts = lines[0].strip().split()
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append([cx, cy, w, h])
                confs.append(1.0)
            else:
                boxes.append([0, 0, 0, 0])
                confs.append(0.0)
        else:
            boxes.append([0, 0, 0, 0])
            confs.append(0.0)

    return np.array(images, dtype=np.float32) / 255.0, np.array(boxes, dtype=np.float32), np.array(confs, dtype=np.float32)

print("Loading training data...")
t0 = time.time()
train_x, train_boxes, train_conf = load_dataset("train")
val_x, val_boxes, val_conf = load_dataset("valid")
print(f"Train: {len(train_x)}, Val: {len(val_x)} (loaded in {time.time()-t0:.1f}s)")
print(f"Positive samples - train: {int(train_conf.sum())}, val: {int(val_conf.sum())}")

# ======================== DATA AUGMENTATION ========================

def augment(image, box, conf):
    # Random horizontal flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        cx, cy, w, h = box[0], box[1], box[2], box[3]
        box = tf.stack([1.0 - cx, cy, w, h])

    # Random brightness/contrast/saturation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, box, conf

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_boxes, train_conf))
train_ds = train_ds.shuffle(len(train_x)).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_boxes, val_conf))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ======================== MODEL ========================

with strategy.scope():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    conf_out = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence")(x)
    box_out = tf.keras.layers.Dense(4, activation="sigmoid", name="bbox")(x)
    model = tf.keras.Model(inputs=base.input, outputs=[conf_out, box_out])

    # Phase 1: frozen backbone
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
        metrics={"confidence": "accuracy"},
    )

model.summary()
print(f"\n{'='*60}")
print(f"Phase 1: Frozen backbone, batch={BATCH_SIZE}, {strategy.num_replicas_in_sync} GPUs")
print(f"{'='*60}")

model.fit(
    train_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    validation_data=val_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    epochs=30, verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]
)

# Phase 2: fine-tune last 50 layers
print(f"\n{'='*60}")
print(f"Phase 2: Fine-tuning top 50 layers")
print(f"{'='*60}")

with strategy.scope():
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
        metrics={"confidence": "accuracy"},
    )

model.fit(
    train_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    validation_data=val_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    epochs=50, verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    ]
)

# Phase 3: fine-tune entire model
print(f"\n{'='*60}")
print(f"Phase 3: Full model fine-tuning")
print(f"{'='*60}")

with strategy.scope():
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss={"confidence": "binary_crossentropy", "bbox": "mse"},
        loss_weights={"confidence": 1.0, "bbox": 5.0},
        metrics={"confidence": "accuracy"},
    )

model.fit(
    train_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    validation_data=val_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    epochs=30, verbose=1,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    ]
)

# ======================== SAVE & EXPORT ========================

save_dir = os.path.expanduser("~/Desktop/19-06TT/coral_ssd")
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, "mobilenet_ssd_ttball.keras"))
print(f"Keras model saved to {save_dir}")

# Evaluate
val_results = model.evaluate(
    val_ds.map(lambda img, box, conf: (img, {"confidence": conf, "bbox": box})),
    verbose=0
)
print(f"Val results: {dict(zip(model.metrics_names, val_results))}")

# Export to TFLite with full int8 quantization using training data as calibration
print("\nExporting to int8 TFLite...")

def representative_dataset():
    for i in range(min(500, len(train_x))):
        yield [np.expand_dims(train_x[i], 0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.float32  # Keep float output for easy post-processing

tflite_model = converter.convert()
tflite_path = os.path.join(save_dir, "mobilenet_ttball_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model: {tflite_path} ({len(tflite_model)/1024/1024:.1f} MB)")

# Compile for Edge TPU
print("\nCompiling for Edge TPU...")
result = subprocess.run(
    ["edgetpu_compiler", "-s", "-o", save_dir, tflite_path],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"edgetpu_compiler error: {result.stderr}")
else:
    edgetpu_path = os.path.join(save_dir, "mobilenet_ttball_int8_edgetpu.tflite")
    if os.path.exists(edgetpu_path):
        sz = os.path.getsize(edgetpu_path)
        print(f"Edge TPU model: {edgetpu_path} ({sz/1024/1024:.1f} MB)")

# Quick TFLite validation
print("\nValidating TFLite model on val set...")
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_dets = interp.get_output_details()
print(f"Input: {inp_det['shape']}, dtype={inp_det['dtype']}")
for i, o in enumerate(out_dets):
    print(f"Output[{i}]: {o['shape']}, dtype={o['dtype']}")

correct = 0
total = min(100, len(val_x))
for i in range(total):
    img = val_x[i]
    if inp_det['dtype'] == np.int8:
        scale, zp = inp_det['quantization']
        img = ((img / scale) + zp).astype(np.int8)
    interp.set_tensor(inp_det['index'], np.expand_dims(img, 0))
    interp.invoke()
    # Find confidence output
    for o in out_dets:
        val = interp.get_tensor(o['index'])[0]
        if val.shape == (1,):
            pred_conf = float(val[0])
            gt = val_conf[i]
            if (pred_conf > 0.5) == (gt > 0.5):
                correct += 1
            break

print(f"TFLite accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print(f"\n{'='*60}")
print("ALL DONE!")
print(f"{'='*60}")
