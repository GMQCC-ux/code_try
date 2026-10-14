# prepare_splits.py
# Put this script anywhere, then edit INPUT_DIR and OUTPUT_DIR before running.
# It handles:
# 1) dataset/images + dataset/masks
# 2) dataset/train/images + dataset/train/masks and dataset/test/images + dataset/test/masks, without val
#
# Output:
# OUTPUT_DIR/train/images, OUTPUT_DIR/train/masks
# OUTPUT_DIR/val/images,   OUTPUT_DIR/val/masks
# OUTPUT_DIR/test/images,  OUTPUT_DIR/test/masks

import os
import random
import shutil
from pathlib import Path

INPUT_DIR = r"D:/datasets/UOA-DR-2017"
OUTPUT_DIR = r"D:/datasets/UOA-DR-2017_split"

SEED = 56

# Used when INPUT_DIR only has images/ and masks/
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Used when INPUT_DIR has train/ and test/ but no val/
VAL_FROM_TRAIN_RATIO = 0.1

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def stem_map(files):
    return {p.stem: p for p in files}


def collect_pairs(images_dir, masks_dir):
    images = list_files(images_dir)
    masks = list_files(masks_dir)

    image_map = stem_map(images)
    mask_map = stem_map(masks)

    common = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))

    if missing_masks:
        print(f"[WARN] {len(missing_masks)} images have no matching mask. Example: {missing_masks[:5]}")
    if missing_images:
        print(f"[WARN] {len(missing_images)} masks have no matching image. Example: {missing_images[:5]}")

    return [(image_map[k], mask_map[k]) for k in common]


def ensure_dirs(root):
    root = Path(root)
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs, out_root, split):
    out_root = Path(out_root)
    for img, mask in pairs:
        shutil.copy2(img, out_root / split / "images" / img.name)
        shutil.copy2(mask, out_root / split / "masks" / mask.name)


def split_list(pairs, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    pairs = pairs[:]
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    return train, val, test


def split_train_val(train_pairs, val_ratio):
    train_pairs = train_pairs[:]
    random.shuffle(train_pairs)

    n_val = max(1, int(len(train_pairs) * val_ratio)) if len(train_pairs) > 1 else 0
    val = train_pairs[:n_val]
    train = train_pairs[n_val:]

    return train, val


def detect_layout(input_dir):
    input_dir = Path(input_dir)

    flat_images = input_dir / "images"
    flat_masks = input_dir / "masks"

    train_images = input_dir / "train" / "images"
    train_masks = input_dir / "train" / "masks"
    val_images = input_dir / "val" / "images"
    val_masks = input_dir / "val" / "masks"
    test_images = input_dir / "test" / "images"
    test_masks = input_dir / "test" / "masks"

    if flat_images.exists() and flat_masks.exists():
        return "flat"

    has_train = train_images.exists() and train_masks.exists()
    has_val = val_images.exists() and val_masks.exists()
    has_test = test_images.exists() and test_masks.exists()

    if has_train and has_val and has_test:
        return "already_split"

    if has_train and has_test and not has_val:
        return "train_test"

    raise RuntimeError(
        "Unsupported dataset layout. Expected one of:\n"
        "1) images/ masks/\n"
        "2) train/images train/masks test/images test/masks\n"
        "3) train/images train/masks val/images val/masks test/images test/masks"
    )


def main():
    random.seed(SEED)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        raise RuntimeError(f"INPUT_DIR does not exist: {input_dir}")

    if output_dir.exists():
        raise RuntimeError(
            f"OUTPUT_DIR already exists: {output_dir}\n"
            "For safety, this script will not delete or overwrite an existing output directory."
        )

    ensure_dirs(output_dir)

    layout = detect_layout(input_dir)
    print(f"[INFO] Detected layout: {layout}")

    if layout == "flat":
        pairs = collect_pairs(input_dir / "images", input_dir / "masks")
        train, val, test = split_list(pairs, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    elif layout == "train_test":
        train_pairs = collect_pairs(input_dir / "train" / "images", input_dir / "train" / "masks")
        test = collect_pairs(input_dir / "test" / "images", input_dir / "test" / "masks")
        train, val = split_train_val(train_pairs, VAL_FROM_TRAIN_RATIO)

    elif layout == "already_split":
        train = collect_pairs(input_dir / "train" / "images", input_dir / "train" / "masks")
        val = collect_pairs(input_dir / "val" / "images", input_dir / "val" / "masks")
        test = collect_pairs(input_dir / "test" / "images", input_dir / "test" / "masks")

    else:
        raise RuntimeError(f"Unknown layout: {layout}")

    copy_pairs(train, output_dir, "train")
    copy_pairs(val, output_dir, "val")
    copy_pairs(test, output_dir, "test")

    print("[DONE] Dataset split completed")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()