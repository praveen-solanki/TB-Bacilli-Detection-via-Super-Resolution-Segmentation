import torch
import cv2 as cv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import re

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
MSK_EXTS = IMG_EXTS
IMG_DIR_HINTS = ("image", "images", "img", "imgs")
MSK_DIR_HINTS = ("mask", "masks", "gt", "ground", "label", "annotation", "annotations")
_SUFFIX_STRIP_RE = re.compile(r"(_?mask|_?gt|_?label|_?ann(otation)?|_?seg)+$", re.IGNORECASE)

def _norm_stem(p: Path) -> str:
    s = p.stem.lower()
    s = _SUFFIX_STRIP_RE.sub("", s)
    return s

def _path_has_hint(path: Path, hints) -> bool:
    name = str(path.as_posix()).lower()
    return any(h in name for h in hints)

def discover_image_mask_pairs(root: Path, split_name: str):
    split_dir = root / split_name
    if not split_dir.exists():
        wanted = split_name.lower().replace(" ", "")
        candidates = [c for c in root.iterdir() if c.is_dir()]
        match = next((c for c in candidates if wanted in c.name.lower().replace(" ", "")), None)
        if not match: raise FileNotFoundError(f"Split folder not found: {split_dir}")
        split_dir = match
    all_files = [p for p in split_dir.rglob("*") if p.is_file()]
    img_files = [p for p in all_files if p.suffix.lower() in IMG_EXTS]
    msk_files = [p for p in all_files if p.suffix.lower() in MSK_EXTS]
    hinted_imgs = [p for p in img_files if _path_has_hint(p.parent, IMG_DIR_HINTS)]
    hinted_msks = [p for p in msk_files if _path_has_hint(p.parent, MSK_DIR_HINTS)]
    if hinted_imgs: img_files = hinted_imgs
    if hinted_msks: msk_files = hinted_msks
    img_map, msk_map = {}, {}
    for p in img_files: img_map.setdefault(_norm_stem(p), []).append(p)
    for p in msk_files: msk_map.setdefault(_norm_stem(p), []).append(p)
    pairs = []
    for key in sorted(img_map.keys()):
        if key in msk_map:
            ip = sorted(img_map[key])[0]
            mp = sorted(msk_map[key])[0]
            pairs.append((ip, mp))
    return pairs

def convert_and_save(pairs, output_path):
    """Reads image pairs, converts them to tensors, and saves to a single file."""
    images = []
    masks = []
    first_img_shape = None

    for ip, mp in tqdm(pairs, desc=f"Converting data for {output_path}"):
        img = cv.imread(str(ip), cv.IMREAD_COLOR)
        if img is None: continue

        # Ensure all images are the same size
        if first_img_shape is None:
            first_img_shape = img.shape
        elif first_img_shape != img.shape:
            print(f"\nSkipping {ip} due to shape mismatch: got {img.shape}, expected {first_img_shape}")
            continue
            
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Convert to RGB
        msk = cv.imread(str(mp), cv.IMREAD_GRAYSCALE)
        
        images.append(img)
        masks.append(msk)

    # Stack all images into a single NumPy array, then convert to a byte tensor
    # to save space. We will convert to float and normalize in the Dataset.
    images_tensor = torch.from_numpy(np.stack(images))
    masks_tensor = torch.from_numpy(np.stack(masks))

    torch.save({
        "images": images_tensor, # Shape: (N, H, W, C)
        "masks": masks_tensor    # Shape: (N, H, W)
    }, output_path)
    print(f"✅ Saved dataset to {output_path}")
    print(f"   - Images tensor shape: {images_tensor.shape}")
    print(f"   - Masks tensor shape: {masks_tensor.shape}")


def main():
    ap = argparse.ArgumentParser(description="Convert an image folder dataset to a single PyTorch tensor file.")
    ap.add_argument("--root", type=str, required=True, help="Path to dataset root that contains split folders")
    ap.add_argument("--train_split", type=str, default="TRAINING SET")
    ap.add_argument("--val_split", type=str, default="VALIDATION SET")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()

    train_pairs = discover_image_mask_pairs(root, args.train_split)
    val_pairs = discover_image_mask_pairs(root, args.val_split)

    convert_and_save(train_pairs, "training_set.pt")
    convert_and_save(val_pairs, "validation_set.pt")

if __name__ == "__main__":
    main()