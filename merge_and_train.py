"""
Merge two Roboflow datasets and train YOLO11s for TT ball detection.
Run on: helium5090@100.97.153.40 (in ~/Desktop/19-06TT/)
"""

import os
import shutil
import glob
from ultralytics import YOLO

WORK = os.path.expanduser("~/Desktop/19-06TT")
DS1 = os.path.join(WORK, "ds1")
DS2 = os.path.join(WORK, "ds2")
MERGED = os.path.join(WORK, "merged_dataset")

def merge_datasets():
    print("=" * 60)
    print("Merging datasets")
    print("=" * 60)

    for split in ['train', 'valid', 'test']:
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(MERGED, split, sub), exist_ok=True)

    total = 0
    for ds_name, ds_path in [("ds1", DS1), ("ds2", DS2)]:
        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(ds_path, split, 'images')
            lbl_dir = os.path.join(ds_path, split, 'labels')

            if not os.path.exists(img_dir):
                continue

            for img in glob.glob(os.path.join(img_dir, '*')):
                stem = os.path.splitext(os.path.basename(img))[0]
                ext = os.path.splitext(img)[1]
                new_name = f"{ds_name}_{stem}"

                shutil.copy2(img, os.path.join(MERGED, split, 'images', new_name + ext))

                lbl = os.path.join(lbl_dir, stem + '.txt')
                if os.path.exists(lbl):
                    shutil.copy2(lbl, os.path.join(MERGED, split, 'labels', new_name + '.txt'))

                total += 1

    # Write data.yaml
    yaml_path = os.path.join(MERGED, "data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {MERGED}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write("nc: 1\n")
        f.write("names: ['ping_pong_ball']\n")

    for split in ['train', 'valid', 'test']:
        n = len(glob.glob(os.path.join(MERGED, split, 'images', '*')))
        print(f"  {split}: {n} images")
    print(f"  Total: {total}")
    print(f"  data.yaml: {yaml_path}")
    return yaml_path


def train(yaml_path):
    print("\n" + "=" * 60)
    print("Training YOLO11s on merged dataset")
    print("=" * 60)

    model = YOLO("yolo11s.pt")

    model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=64,
        name="ttball_merged_v1",
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
        flipud=0.0,
        workers=8,
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=25,
        val=True,
        plots=True,
        seed=42,
        device=[0, 1],
    )

    best = os.path.join(WORK, "runs", "ttball_merged_v1", "weights", "best.pt")
    print(f"\nBest weights: {best}")
    return best


if __name__ == '__main__':
    yaml_path = merge_datasets()
    best = train(yaml_path)
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {best}")
    print(f"{'='*60}")
