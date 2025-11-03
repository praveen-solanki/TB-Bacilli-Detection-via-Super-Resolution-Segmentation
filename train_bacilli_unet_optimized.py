# train_bacilli_unet_optimized_windows_safe.py
import argparse
import os
from pathlib import Path
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import cv2 as cv
import kornia.augmentation as K

# -----------------------------
# Windows sharing-strategy fix
# -----------------------------
# On Windows, the default PyTorch sharing strategy may cause "Couldn't open shared file mapping"
# errors when many workers / large tensors are used. Use 'file_system' strategy to avoid
# some kernel handle/name mapping limits.
# Note: calling this should happen before creating DataLoader workers.
try:
    import torch.multiprocessing as _mp
    # Only change strategy if available and we're on Windows
    if sys.platform.startswith("win"):
        try:
            _mp.set_sharing_strategy("file_system")
            print("✔ PyTorch multiprocessing sharing strategy set to 'file_system' (Windows).")
        except Exception as e:
            print("⚠️ Could not set sharing strategy to 'file_system':", e)
except Exception:
    pass

# -----------------------------
# U-Net (unchanged)
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
        return self.outc(u1)

# -----------------------------
# PreloadedTensorDataset (unchanged)
# -----------------------------
class PreloadedTensorDataset(Dataset):
    def __init__(self, pt_file_path):
        print(f"📦 Loading dataset from {pt_file_path} ...")
        data = torch.load(pt_file_path)
        self.images = data["images"].permute(0, 3, 1, 2).float()
        self.masks = data["masks"].unsqueeze(1).float()
        assert len(self.images) == len(self.masks), "Mismatch between image/mask count!"
        self.image_height, self.image_width = self.images.shape[2], self.images.shape[3]
        print(f"✅ Loaded {len(self.images)} samples, each {self.image_height}x{self.image_width}")
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx] / 255.0, self.masks[idx] / 255.0

# -----------------------------
# Losses & metrics (unchanged)
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(2,3)) + self.eps
        den = (probs*probs).sum(dim=(2,3)) + (targets*targets).sum(dim=(2,3)) + self.eps
        return 1.0 - (num/den).mean()

def iou_and_dice(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    inter = (preds*targets).sum(dim=(2,3))
    union = (preds + targets - preds*targets).sum(dim=(2,3)) + eps
    iou = (inter + eps) / union
    dice = (2*inter + eps) / (preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps)
    return iou.mean().item(), dice.mean().item()

# -----------------------------
# Train/validate/predict (unchanged)
# -----------------------------
def train_one_epoch(model, loader, optimizer, bce, dl, gpu_aug=None, scaler=None, device="cuda"):
    model.train()
    metrics = {"loss":0.0, "iou":0.0, "dice":0.0}
    for imgs, msks in tqdm(loader, desc="Train", leave=False):
        imgs, msks = imgs.to(device), msks.to(device)
        if gpu_aug:
            stacked = torch.cat((imgs, msks), dim=1)
            stacked_aug = gpu_aug(stacked)
            imgs, msks = torch.split(stacked_aug, [3,1], dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss = 0.5 * bce(logits, msks) + 0.5 * dl(logits, msks)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        with torch.no_grad():
            iou, dice = iou_and_dice(logits, msks)
        metrics["loss"] += loss.item(); metrics["iou"] += iou; metrics["dice"] += dice
    n = len(loader); return {k:v/max(1,n) for k,v in metrics.items()}

@torch.no_grad()
def validate(model, loader, bce, dl, device="cuda"):
    model.eval()
    out={"loss":0.0,"iou":0.0,"dice":0.0}
    for imgs, msks in tqdm(loader, desc="Valid", leave=False):
        imgs, msks = imgs.to(device), msks.to(device)
        logits = model(imgs)
        loss = 0.5 * bce(logits, msks) + 0.5 * dl(logits, msks)
        iou, dice = iou_and_dice(logits, msks)
        out["loss"] += loss.item(); out["iou"] += iou; out["dice"] += dice
    n = len(loader); return {k:v/max(1,n) for k,v in out.items()}

@torch.no_grad()
def predict_folder(model, ckpt, in_dir, out_dir, train_img_size, device="cuda", thresh=0.5):
    model_to_load = getattr(model, "_orig_mod", model)
    ckpt_data = torch.load(ckpt, map_location=device)
    model_to_load.load_state_dict(ckpt_data["model"])
    model.eval()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(in_dir)
    imgs = sorted(in_dir.glob("*.bmp")) + sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
    h_train, w_train = train_img_size
    for p in tqdm(imgs, desc="Predict"):
        img = cv.imread(str(p), cv.IMREAD_COLOR)
        if img is None: continue
        h_orig, w_orig = img.shape[:2]
        img_r = cv.resize(img, (w_train, h_train), interpolation=cv.INTER_AREA)
        img_rgb = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(np.transpose(img_rgb, (2,0,1))).unsqueeze(0).to(device)
        logit = model(x)
        prob = torch.sigmoid(logit)[0,0].cpu().numpy()
        pred = (prob >= thresh).astype(np.uint8) * 255
        pred = cv.resize(pred, (w_orig, h_orig), interpolation=cv.INTER_NEAREST)
        cv.imwrite(str(out_dir / (p.stem + "_pred.png")), pred)

# -----------------------------
# Main (with robust DataLoader creation)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Optimized U-Net training for bacilli segmentation (Windows-safe DataLoader)")
    ap.add_argument("--train_data", type=str, required=True, help="Path to training_set.pt")
    ap.add_argument("--val_data", type=str, required=True, help="Path to validation_set.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=str, default="./runs_bacilli")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--predict_after", type=str, default="")
    args = ap.parse_args()

    # device selection
    if torch.cuda.is_available(): device="cuda"
    elif torch.backends.mps.is_available(): device="mps"; args.amp=False
    else: device="cpu"; args.amp=False
    print("Using device:", device)
    torch.backends.cudnn.benchmark = True

    # load datasets
    train_ds = PreloadedTensorDataset(args.train_data)
    val_ds = PreloadedTensorDataset(args.val_data)

    # Try to create DataLoaders with requested workers; fallback to num_workers=0 on failure
    num_workers = args.workers
    pin_memory = (device == "cuda")
    try:
        print(f"Creating DataLoaders with num_workers={num_workers}, pin_memory={pin_memory} ...")
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=max(1, args.batch*2), shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    except Exception as e:
        warnings.warn(f"DataLoader worker creation failed (workers={num_workers}, pin_memory={pin_memory}). "
                      f"Falling back to num_workers=0 and pin_memory=False. Error: {e}")
        num_workers = 0
        pin_memory = False
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=max(1, args.batch*2), shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    # model
    model = UNet(in_ch=3, out_ch=1, base=32).to(device)
    try:
        model = torch.compile(model)
        print("Model compiled.")
    except Exception as e:
        print("torch.compile failed, continuing without compile:", e)

    # GPU augmentations
    gpu_aug = None
    if not args.no_augment and device == "cuda":
        gpu_aug = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=15.0, p=0.3)
        ).to(device)
        print("Using Kornia GPU augmentations.")

    # losses & optimizer
    bce = nn.BCEWithLogitsLoss()
    dl = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device == "cuda" else None

    os.makedirs(args.outdir, exist_ok=True)
    best_dice = -1.0
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model_to_load = getattr(model, "_orig_mod", model)
        model_to_load.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_val_dice", -1.0)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, best_dice={best_dice:.4f})")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr = train_one_epoch(model, train_dl, optimizer, bce, dl, gpu_aug, scaler, device)
        va = validate(model, val_dl, bce, dl, device)
        print(f"Train: loss={tr['loss']:.4f}, IoU={tr['iou']:.4f}, Dice={tr['dice']:.4f}")
        print(f"Valid: loss={va['loss']:.4f}, IoU={va['iou']:.4f}, Dice={va['dice']:.4f}")

        ckpt = {"epoch": epoch, "model": getattr(model, "_orig_mod", model).state_dict(),
                "optimizer": optimizer.state_dict(), "best_val_dice": best_dice}
        torch.save(ckpt, Path(args.outdir) / "last.pt")
        if va["dice"] > best_dice:
            best_dice = va["dice"]
            torch.save(ckpt, Path(args.outdir) / "best.pt")
            print(f"✅ New best Dice {best_dice:.4f} — model saved.")

    # optional prediction
    if args.predict_after:
        best_path = Path(args.outdir) / "best.pt"
        if best_path.exists():
            print(f"\nRunning predictions using {best_path} …")
            predict_folder(model, str(best_path), args.predict_after,
                           Path(args.outdir) / "preds",
                           train_img_size=(train_ds.image_height, train_ds.image_width),
                           device=device)
        else:
            print("Best checkpoint not found; skipping prediction.")

if __name__ == "__main__":
    main()
