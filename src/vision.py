import os
import argparse
import random
from pathlib import Path
from shutil import copy2
from PIL import Image
from glob import glob

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def collect_images(raw_dir):
    """
    Collect all gesture class folders from the LeapGestRecog dataset.
    Expected structure:
    raw_dir/leapGestRecog/subject_id/gesture_name/*.png
    """
    class_to_files = {}
    base_path = os.path.join(raw_dir, "leapGestRecog")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"❌ LeapGestRecog folder not found at {base_path}")

    # Traverse all subject folders (00, 01, 02, ...)
    for subject_path in glob(os.path.join(base_path, "*")):
        if not os.path.isdir(subject_path):
            continue

        # Each gesture folder inside subject
        for gesture_path in glob(os.path.join(subject_path, "*")):
            if not os.path.isdir(gesture_path):
                continue

            gesture_name = os.path.basename(gesture_path)
            if gesture_name not in class_to_files:
                class_to_files[gesture_name] = []

            # Collect all images in gesture folder
            class_to_files[gesture_name].extend(glob(os.path.join(gesture_path, "*.png")))

    if not class_to_files:
        raise FileNotFoundError(f"❌ No gesture folders found in {base_path}")

    print(f" Found {len(class_to_files)} gesture classes.")
    total_images = sum(len(v) for v in class_to_files.values())
    print(f" Total images collected: {total_images}")
    print(f" Classes: {list(class_to_files.keys())[:10]}")
    return class_to_files


def split_and_copy(class_to_files, out_dir, img_size=128, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split data into train/val/test sets and resize images.
    """
    random.seed(seed)
    out_dir = Path(out_dir)
    train_dir = out_dir / 'train'
    val_dir = out_dir / 'val'
    test_dir = out_dir / 'test'
    for d in [train_dir, val_dir, test_dir]:
        ensure_dir(d)

    print("\n Splitting dataset and resizing images...")

    for cls, files in class_to_files.items():
        if not files:
            continue
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            target_class_dir = out_dir / split_name / cls
            ensure_dir(target_class_dir)
            for idx, fp in enumerate(split_files):
                try:
                    img = Image.open(fp).convert('RGB')
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    out_name = f"{cls}_{idx:05d}.png"
                    img.save(target_class_dir / out_name)
                except Exception as e:
                    print(f"⚠️ Skipping {fp}: {e}")

    print("\n✅ Dataset successfully prepared at:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Path to raw dataset folder")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Where to save processed data")
    parser.add_argument("--img_size", type=int, default=128, help="Resize images to this size (square)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_to_files = collect_images(args.raw_dir)
    if not class_to_files:
        raise RuntimeError("No images found in raw dataset directory. Check structure.")
    split_and_copy(class_to_files, args.out_dir, args.img_size, args.train_ratio, args.val_ratio, args.seed)


