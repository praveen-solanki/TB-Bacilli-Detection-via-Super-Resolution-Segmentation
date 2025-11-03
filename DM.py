# train_bacilli_unet.py
import argparse
import os
import random
from pathlib import Path
import re

import cv2 as cv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utilities & matching
# -----------------------------
SPLIT_DIR_NAMES = ["TRAINING SET", "VALIDATION SET", "TESTING SET"]

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
        if not match:
            raise FileNotFoundError(f"Split folder not found: {split_dir}")
        print(f"✔ Matched split '{split_name}' -> '{match.name}'")
        split_dir = match

    all_files = [p for p in split_dir.rglob("*") if p.is_file()]
    img_files = [p for p in all_files if p.suffix.lower() in IMG_EXTS]
    msk_files = [p for p in all_files if p.suffix.lower() in MSK_EXTS]

    hinted_imgs = [p for p in img_files if _path_has_hint(p.parent, IMG_DIR_HINTS)]
    hinted_msks = [p for p in msk_files if _path_has_hint(p.parent, MSK_DIR_HINTS)]
    if hinted_imgs:
        img_files = hinted_imgs
    if hinted_msks:
        msk_files = hinted_msks

    img_map, msk_map = {}, {}
    for p in img_files:
        img_map.setdefault(_norm_stem(p), []).append(p)
    for p in msk_files:
        msk_map.setdefault(_norm_stem(p), []).append(p)

    pairs = []
    for key in sorted(img_map.keys()):
        if key in msk_map:
            ip = sorted(img_map[key])[0]
            mp = sorted(msk_map[key])[0]
            pairs.append((ip, mp))

    print(f"[DISCOVERY] Split: {split_dir}")
    print(f"  image candidates: {len(img_files)}, mask candidates: {len(msk_files)}, paired: {len(pairs)}")
    if len(pairs) == 0:
        img_keys = list(sorted(img_map.keys()))[:10]
        msk_keys = list(sorted(msk_map.keys()))[:10]
        print("  sample image stems:", img_keys)
        print("  sample mask stems :", msk_keys)
        raise RuntimeError(f"No IMAGE/MASK pairs found under {split_dir}. "
                           "Check naming; matcher expects stems to match after stripping _mask/_gt/etc.")
    return pairs

# -----------------------------
# Lightweight U-Net
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.down2 = DoubleConv(base, base*2)
        self.down3 = DoubleConv(base*2, base*4)
        self.down4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.conv4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        bn = self.bottleneck(self.pool(d4))
        u4 = self.up4(bn); u4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4); u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3); u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2); u1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.outc(u1)  # logits

# -----------------------------
# Preprocessing helpers (no augmentation)
# -----------------------------
def to_tensor(img_bgr, msk01):
    # convert BGR->RGB, float32 [0,1], channel-first
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    msk = msk01.astype(np.float32)[None, ...]
    return torch.from_numpy(img), torch.from_numpy(msk)

# -----------------------------
# Dataset with optional caching (no resizing / cropping / flips / color jitter)
# -----------------------------
class DDS3SegDataset(Dataset):
    @staticmethod
    def imread_bgr(path: Path):
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return img

    @staticmethod
    def read_mask_binary(path: Path):
        m = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(path)
        if m.ndim == 3:
            m = cv.cvtColor(m, cv.COLOR_BGR2GRAY)
        return (m > 0).astype(np.uint8)

    def __init__(self, pairs, cache=False):
        """
        pairs: list of (image_path, mask_path)
        cache: if True, load everything into memory as tensors (img_t, msk_t)
        """
        self.pairs = pairs
        self.cache = cache
        self.cache_data = None
        if self.cache:
            print("[CACHE] Loading dataset into memory (this may use a lot of RAM)...")
            cd = []
            for ip, mp in tqdm(self.pairs, desc="Caching dataset"):
                img = self.imread_bgr(ip)
                msk = self.read_mask_binary(mp)
                img_t, msk_t = to_tensor(img, msk)
                cd.append((img_t, msk_t))
            self.cache_data = cd
            print(f"[CACHE] Cached {len(self.cache_data)} items.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache and self.cache_data is not None:
            return self.cache_data[idx]
        ip, mp = self.pairs[idx]
        img = self.imread_bgr(ip)
        msk = self.read_mask_binary(mp)
        img_t, msk_t = to_tensor(img, msk)
        return img_t, msk_t

# -----------------------------
# Losses & metrics
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs*targets).sum(dim=(2,3)) + self.eps
        den = (probs*probs).sum(dim=(2,3)) + (targets*targets).sum(dim=(2,3)) + self.eps
        dice = num/den
        return 1.0 - dice.mean()

def iou_and_dice(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    inter = (preds*targets).sum(dim=(2,3))
    union = (preds + targets - preds*targets).sum(dim=(2,3)) + eps
    iou = (inter + eps) / union
    dice = (2*inter + eps) / (preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps)
    return iou.mean().item(), dice.mean().item()

# -----------------------------
# Collate: pad batch to same HxW, rounding up to multiple of 16
# -----------------------------
def _ceil_to_multiple(x, m):
    return ((x + m - 1) // m) * m

def pad_collate(batch):
    """
    batch: list of (img_t [C,H,W], msk_t [1,H,W])
    Pads each sample on right/bottom so that all have same H/W and those dims are multiples of 16.
    Returns stacked tensors (N, C, H_pad, W_pad), (N, 1, H_pad, W_pad)
    """
    imgs, msks = zip(*batch)
    # find max H/W
    max_h = max([i.shape[1] for i in imgs])
    max_w = max([i.shape[2] for i in imgs])
    H_pad = _ceil_to_multiple(max_h, 16)
    W_pad = _ceil_to_multiple(max_w, 16)

    imgs_p = []
    msks_p = []
    for img, msk in zip(imgs, msks):
        c, h, w = img.shape
        pad_h = H_pad - h
        pad_w = W_pad - w
        # F.pad takes pad as (left, right, top, bottom) for 2D, but for (C,H,W) we pad on last two dims:
        img_p = F.pad(img, (0, pad_w, 0, pad_h))  # pads width then height
        msk_p = F.pad(msk, (0, pad_w, 0, pad_h))
        imgs_p.append(img_p)
        msks_p.append(msk_p)
    imgs_b = torch.stack(imgs_p, dim=0)
    msks_b = torch.stack(msks_p, dim=0)
    return imgs_b, msks_b

# -----------------------------
# Train / validate
# -----------------------------
def train_one_epoch(model, loader, optimizer, bce, dl, scaler=None, device="cuda"):
    model.train()
    running = {"loss": 0.0, "iou": 0.0, "dice": 0.0}
    for imgs, msks in tqdm(loader, desc="Train", leave=False):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = (scaler is not None)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = 0.5 * bce(logits, msks) + 0.5 * dl(logits, msks)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs)
            loss = 0.5 * bce(logits, msks) + 0.5 * dl(logits, msks)
            loss.backward(); optimizer.step()

        with torch.no_grad():
            iou, dice = iou_and_dice(logits, msks)
        running["loss"] += loss.item()
        running["iou"]  += iou
        running["dice"] += dice

    n = len(loader)
    return {k: v / max(1, n) for k, v in running.items()}

@torch.no_grad()
def validate(model, loader, bce, dl, device="cuda"):
    model.eval()
    out = {"loss": 0.0, "iou": 0.0, "dice": 0.0}
    for imgs, msks in tqdm(loader, desc="Valid", leave=False):
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        loss = 0.5 * bce(logits, msks) + 0.5 * dl(logits, msks)
        iou, dice = iou_and_dice(logits, msks)
        out["loss"] += loss.item()
        out["iou"]  += iou
        out["dice"] += dice
    n = len(loader)
    return {k: v / max(1, n) for k, v in out.items()}

# -----------------------------
# Prediction helper (no resizing; pads to multiple of 16 and then unpads)
# -----------------------------
@torch.no_grad()
def predict_folder(model, ckpt, in_dir, out_dir, device="cuda", thresh=0.5):
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(in_dir)
    imgs = sorted(list(in_dir.glob("*.bmp")) + list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")))
    for p in tqdm(imgs, desc="Predict"):
        img = DDS3SegDataset.imread_bgr(p)
        h, w = img.shape[:2]
        x, _ = to_tensor(img, (np.zeros((h, w), np.uint8)))
        # pad to multiple of 16
        H_pad = _ceil_to_multiple(h, 16)
        W_pad = _ceil_to_multiple(w, 16)
        x_p = F.pad(x, (0, W_pad - w, 0, H_pad - h)).unsqueeze(0).to(device)
        logit = model(x_p)
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
        prob = prob[:h, :w]  # unpad
        pred = (prob >= thresh).astype(np.uint8) * 255
        cv.imwrite(str(out_dir / (p.stem + "_pred.png")), pred)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train U-Net (no resizing/augmentation; optional cache)")
    ap.add_argument("--root", type=str, required=True, help="Path to dataset root that contains split folders")
    ap.add_argument("--train_split", type=str, default="TRAINING SET", choices=SPLIT_DIR_NAMES)
    ap.add_argument("--val_split", type=str, default="VALIDATION SET", choices=SPLIT_DIR_NAMES)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (set higher on Linux/CUDA machines)")
    ap.add_argument("--outdir", type=str, default="./runs_bacilli")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    ap.add_argument("--predict_after", type=str, default="", help="Optional folder of images to run predictions on after training")
    ap.add_argument("--cache", action="store_true", help="Cache preprocessed images in memory (may use lots of RAM)")
    ap.add_argument("--base", type=int, default=32, help="Base channel width for UNet (try 16 for faster/smaller)")
    args = ap.parse_args()

    # Device pick (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        if args.amp:
            print("Note: AMP is CUDA-only here; ignoring --amp on MPS.")
            args.amp = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = "cpu"
        if args.amp:
            print("Note: AMP is CUDA-only here; ignoring --amp on CPU.")
            args.amp = False
    print("Using device:", device)

    # cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    root = Path(args.root).expanduser().resolve()
    print("Using dataset root:", root)

    train_pairs = discover_image_mask_pairs(root, args.train_split)
    val_pairs   = discover_image_mask_pairs(root, args.val_split)

    train_ds = DDS3SegDataset(train_pairs, cache=args.cache)
    val_ds   = DDS3SegDataset(val_pairs,   cache=args.cache)

    # pin memory only on CUDA
    use_pin = (device == "cuda")
    num_workers = args.workers if device == "cuda" else args.workers  # allow using workers also on CPU if desired

    print("Sample pair:", train_pairs[0])
    # quick check of reading
    print("imread_bgr via class OK:", DDS3SegDataset.imread_bgr(train_pairs[0][0]).shape)
    print("read_mask_binary via class OK:", DDS3SegDataset.read_mask_binary(train_pairs[0][1]).shape)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=num_workers, pin_memory=use_pin, drop_last=True,
                          collate_fn=pad_collate)
    val_dl   = DataLoader(val_ds, batch_size=max(1, args.batch // 2), shuffle=False,
                          num_workers=num_workers, pin_memory=use_pin,
                          collate_fn=pad_collate)

    model = UNet(in_ch=3, out_ch=1, base=args.base).to(device)

    bce = nn.BCEWithLogitsLoss()
    dl  = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    os.makedirs(args.outdir, exist_ok=True)
    start_epoch, best_val_dice = 0, -1.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", -1.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, best_val_dice={best_val_dice:.4f}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr = train_one_epoch(model, train_dl, optimizer, bce, dl, scaler, device)
        va = validate(model, val_dl, bce, dl, device)
        print(f"Train: loss={tr['loss']:.4f}, IoU={tr['iou']:.4f}, Dice={tr['dice']:.4f}")
        print(f"Valid: loss={va['loss']:.4f}, IoU={va['iou']:.4f}, Dice={va['dice']:.4f}")

        # save last
        last_path = Path(args.outdir) / "last.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_dice": best_val_dice
        }, last_path)

        # save best
        if va["dice"] > best_val_dice:
            best_val_dice = va["dice"]
            best_path = Path(args.outdir) / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_dice": best_val_dice
            }, best_path)
            print(f"✅ New best Dice {best_val_dice:.4f} — saved {best_path}")

    if args.predict_after:
        best_path = Path(args.outdir) / "best.pt"
        if best_path.exists():
            print(f"\nRunning predictions using {best_path} …")
            predict_folder(model, str(best_path), args.predict_after,
                           Path(args.outdir) / "preds", device=device)
        else:
            print("Best checkpoint not found; skipping prediction.")

if __name__ == "__main__":
    main()
