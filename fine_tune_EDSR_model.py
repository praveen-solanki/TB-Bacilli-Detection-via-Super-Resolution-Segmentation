"""
finetune_edsr_simple.py

Single-file script to FINE-TUNE an EDSR x4 model on your TB microscopy dataset (paired LR/HR).
Place your dataset like:
  TB_dataset/
    HR/
    LR/

Edit the CONFIG dict below and run:
  - To train/fine-tune: set MODE = "train" and run the file.
  - To predict: set MODE = "predict" and run the file (provide checkpoint/input/output in CONFIG).

This file intentionally avoids argparse; change values in CONFIG.
"""

import os
import math
import random
import time
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

# ----------------------------
# CONFIG — put all hyperparams & paths here
# ----------------------------
CONFIG = {
    # Mode: "train" or "predict"
    #"MODE": "train",
    "MODE": "predict",

    # Dataset & IO
    "DATA_DIR": "./TB_dataset",      # must contain HR/ and LR/ subfolders
    "SAVE_DIR": "./checkpoints",

    # Pretrained checkpoint for fine-tuning (required for MODE="train")
    # Can also point to a .pth that contains {'state_dict': ...} or a plain state_dict
    "PRETRAINED_PATH": "C:\\Users\\praveen solanki\\Desktop\\PGP\\models\\EDSR_x4.pt",

    # If you want to resume from a previous fine-tune checkpoint, put its path here.
    # If both PRETRAINED_PATH and RESUME_CHECKPOINT are provided, RESUME_CHECKPOINT takes precedence.
    "RESUME_CHECKPOINT": "",

    # Training hyperparams
    "EPOCHS": 50,
    "BATCH_SIZE": 8,
    "PATCH_SIZE_HR": 128,   # HR patch size (will use LR patch size = PATCH_SIZE_HR // SCALE)
    "LR": 1e-4,
    "LR_STEP": 30,
    "LR_GAMMA": 0.5,
    "PRINT_EVERY": 50,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,

    # Model params
    "SCALE": 4,
    "N_RESBLOCKS": 32,
    "N_FEATS": 128,

    # Data augmentation & split
    "AUGMENT": True,
    "VALIDATION_SPLIT": 0.05,  # fraction from train used for validation if no separate val folder
    "SEED": 1234,

    # Device: "cuda" or "cpu" (auto-detect)
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # Grayscale (microscopy often single-channel). If True, model/input/output use 1 channel.
    "GRAYSCALE": False,

    # Prediction settings (used when MODE="predict")
    "PREDICT_CHECKPOINT": 'C:\\Users\\praveen solanki\\Desktop\\PGP\\models\\2nd\\edsr_ft_best.pth',
    "INPUT_IMAGE": "C:\\Users\\praveen solanki\\Desktop\\PGP\EDSR\\tb1.jpg",   # single LR image for prediction
    "OUTPUT_IMAGE": "C:\\Users\\praveen solanki\\Desktop\\PGP\\EDSR\\tb_out.jpg",
}

# ----------------------------
# Model: EDSR (channel-flexible)
# ----------------------------
class ResidualBlockNoBN(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res * self.res_scale

class EDSR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale=4, n_resblocks=16, n_feats=64, rgb_range=1.0):
        super().__init__()
        self.scale = scale
        self.rgb_range = rgb_range
        self.conv_head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        body = [ResidualBlockNoBN(n_feats) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*body)
        self.conv_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        # tail with pixelshuffle blocks
        tail = []
        n_upscale = int(math.log2(scale))
        assert 2**n_upscale == scale, "scale must be power of 2"
        for _ in range(n_upscale):
            tail.append(nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1))
            tail.append(nn.PixelShuffle(2))
            tail.append(nn.ReLU(inplace=True))
        tail.append(nn.Conv2d(n_feats, out_channels, 3, 1, 1))
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = x * self.rgb_range
        x = self.conv_head(x)
        res = self.body(x)
        res = self.conv_body(res)
        x = x + res
        x = self.tail(x)
        x = torch.clamp(x / self.rgb_range, 0.0, 1.0)
        return x

# ----------------------------
# Dataset for paired LR-HR images
# ----------------------------
class PairedTBDataset(Dataset):
    def __init__(self, data_dir, mode='train', scale=4, patch_size_hr=128, augment=True, grayscale=False):
        self.hr_dir = os.path.join(data_dir, "HR")
        self.lr_dir = os.path.join(data_dir, "LR")
        assert os.path.isdir(self.hr_dir) and os.path.isdir(self.lr_dir), f"HR and LR dirs required in {data_dir}"
        hr_files = sorted(glob(os.path.join(self.hr_dir, "*")))
        lr_files = sorted(glob(os.path.join(self.lr_dir, "*")))
        hr_map = {os.path.basename(p): p for p in hr_files}
        lr_map = {os.path.basename(p): p for p in lr_files}
        common = sorted(set(hr_map.keys()) & set(lr_map.keys()))
        if len(common) == 0:
            raise RuntimeError("No matching filenames between HR/ and LR/. Ensure filenames match.")
        self.pairs = [(lr_map[k], hr_map[k]) for k in common]
        self.mode = mode
        self.scale = scale
        self.patch_size_hr = patch_size_hr
        self.augment = augment if mode == 'train' else False
        self.grayscale = grayscale

    def __len__(self):
        return len(self.pairs)

    def random_crop(self, lr_img, hr_img):
        lr_w, lr_h = lr_img.size
        lr_ps = self.patch_size_hr // self.scale
        if lr_w < lr_ps or lr_h < lr_ps:
            lr_img = lr_img.resize((max(lr_ps, lr_w), max(lr_ps, lr_h)), Image.BICUBIC)
            hr_img = hr_img.resize((max(lr_ps, lr_w)*self.scale, max(lr_ps, lr_h)*self.scale), Image.BICUBIC)
            lr_w, lr_h = lr_img.size
        x = random.randint(0, lr_w - lr_ps)
        y = random.randint(0, lr_h - lr_ps)
        lr_patch = lr_img.crop((x, y, x + lr_ps, y + lr_ps))
        hr_patch = hr_img.crop((x * self.scale, y * self.scale, (x + lr_ps) * self.scale, (y + lr_ps) * self.scale))
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr_img = Image.open(lr_path).convert("L" if self.grayscale else "RGB")
        hr_img = Image.open(hr_path).convert("L" if self.grayscale else "RGB")
        if self.mode == 'train':
            lr_patch, hr_patch = self.random_crop(lr_img, hr_img)
            if self.augment:
                if random.random() < 0.5:
                    lr_patch = TF.hflip(lr_patch); hr_patch = TF.hflip(hr_patch)
                if random.random() < 0.5:
                    lr_patch = TF.vflip(lr_patch); hr_patch = TF.vflip(hr_patch)
                rot = random.choice([0, 1, 2, 3])
                if rot:
                    lr_patch = lr_patch.rotate(90 * rot)
                    hr_patch = hr_patch.rotate(90 * rot)
            lr_tensor = TF.to_tensor(lr_patch)
            hr_tensor = TF.to_tensor(hr_patch)
        else:
            lr_tensor = TF.to_tensor(lr_img)
            hr_tensor = TF.to_tensor(hr_img)
        return lr_tensor, hr_tensor

# ----------------------------
# Utilities
# ----------------------------
def save_checkpoint(state, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    torch.save(state, path)

def load_state_dict_flexible(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get('state_dict', ckpt)
    # Try strict=True first; if mismatch, load with strict=False
    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint with strict=True")
    except Exception as e:
        print("strict=True load failed (likely channel or minor key mismatch). Loading with strict=False. Err:", e)
        model.load_state_dict(state, strict=False)
    return ckpt.get('epoch', 0)

def calc_psnr(sr, hr, shave_border=0):
    # expects tensors in [0,1], shapes (B,C,H,W)
    diff = sr - hr
    if shave_border:
        diff = diff[:, :, shave_border:-shave_border, shave_border:-shave_border]
    mse = torch.mean(diff ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()

# ----------------------------
# Training loop
# ----------------------------
def train(config):
    device = torch.device(config["DEVICE"])
    in_c = 1 if config["GRAYSCALE"] else 3
    out_c = in_c

    # model
    model = EDSR(in_channels=in_c, out_channels=out_c, scale=config["SCALE"],
                 n_resblocks=config["N_RESBLOCKS"], n_feats=config["N_FEATS"])
    model = model.to(device)

    # load pretrained (for fine-tuning)
    if config["RESUME_CHECKPOINT"]:
        print("Resuming training from:", config["RESUME_CHECKPOINT"])
        start_epoch = load_state_dict_flexible(model, config["RESUME_CHECKPOINT"])
        start_epoch = int(start_epoch) + 1
    else:
        if not config["PRETRAINED_PATH"]:
            raise ValueError("Provide PRETRAINED_PATH in CONFIG for fine-tuning.")
        print("Loading pretrained weights (fine-tune base) from:", config["PRETRAINED_PATH"])
        load_state_dict_flexible(model, config["PRETRAINED_PATH"])
        start_epoch = 1

    # dataset and loaders
    ds_train = PairedTBDataset(config["DATA_DIR"], mode='train', scale=config["SCALE"],
                               patch_size_hr=config["PATCH_SIZE_HR"], augment=config["AUGMENT"], grayscale=config["GRAYSCALE"])

    # Build train/val split
    dataset_size = len(ds_train)
    indices = list(range(dataset_size))
    random.seed(config["SEED"])
    random.shuffle(indices)
    val_split = int(math.floor(config["VALIDATION_SPLIT"] * dataset_size))
    if val_split < 1:
        val_split = max(1, int(0.05 * dataset_size))  # ensure at least 1
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(ds_train, batch_size=config["BATCH_SIZE"], sampler=train_sampler,
                              num_workers=config["NUM_WORKERS"], pin_memory=config["PIN_MEMORY"], drop_last=True)
    val_loader = DataLoader(ds_train, batch_size=1, sampler=val_sampler,
                            num_workers=1, pin_memory=config["PIN_MEMORY"])

    # optimizer, loss, scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["LR"], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["LR_STEP"], gamma=config["LR_GAMMA"])
    criterion = nn.L1Loss()

    best_psnr = 0.0

    for epoch in range(start_epoch, config["EPOCHS"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i, (lr, hr) in enumerate(train_loader, 1):
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % config["PRINT_EVERY"] == 0 or i == len(train_loader):
                print(f"Epoch[{epoch}] Iter[{i}/{len(train_loader)}] Loss: {loss.item():.6f}")
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        t1 = time.time()
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.6f}. Time: {(t1-t0):.1f}s. LR: {optimizer.param_groups[0]['lr']:.2e}")

        # validation
        model.eval()
        psnr_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr = lr.to(device)
                hr = hr.to(device)
                sr = model(lr)
                psnr = calc_psnr(sr, hr, shave_border=config["SCALE"])
                psnr_sum += psnr
                n_val += 1
        mean_psnr = psnr_sum / max(1, n_val)
        print(f"Validation PSNR: {mean_psnr:.4f} dB on {n_val} images")

        # save checkpoints
        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr': mean_psnr,
            'config': config
        }
        save_checkpoint(ckpt, config["SAVE_DIR"], f"edsr_ft_epoch{epoch}.pth")
        save_checkpoint(ckpt, config["SAVE_DIR"], "edsr_ft_last.pth")
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            save_checkpoint(ckpt, config["SAVE_DIR"], "edsr_ft_best.pth")
            print("Saved new best checkpoint (psnr improved).")

    print("Training complete. Best PSNR:", best_psnr)

# ----------------------------
# Predict single image
# ----------------------------
def predict_single(config):
    device = torch.device(config["DEVICE"])
    in_c = 1 if config["GRAYSCALE"] else 3
    out_c = in_c
    model = EDSR(in_channels=in_c, out_channels=out_c, scale=config["SCALE"],
                 n_resblocks=config["N_RESBLOCKS"], n_feats=config["N_FEATS"])
    model = model.to(device)

    ckpt_path = config["PREDICT_CHECKPOINT"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Predict checkpoint not found: {ckpt_path}")
    print("Loading checkpoint for prediction:", ckpt_path)
    load_state_dict_flexible(model, ckpt_path)

    model.eval()
    img = Image.open(config["INPUT_IMAGE"]).convert("L" if config["GRAYSCALE"] else "RGB")
    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = model(lr_tensor)
    sr = sr.squeeze(0).cpu().clamp(0,1)
    to_pil = transforms.ToPILImage()
    out_pil = to_pil(sr)
    out_pil.save(config["OUTPUT_IMAGE"])
    print("Saved SR image to:", config["OUTPUT_IMAGE"])

# ----------------------------
# Entry point: pick mode from CONFIG
# ----------------------------
if __name__ == "__main__":
    cfg = CONFIG
    random.seed(cfg["SEED"])
    torch.manual_seed(cfg["SEED"])
    if cfg["MODE"] == "train":
        print("Starting fine-tune EDSR (mode=train). Double-check CONFIG at top.")
        train(cfg)
    elif cfg["MODE"] == "predict":
        print("Running single-image prediction (mode=predict).")
        predict_single(cfg)
    else:
        raise ValueError("Unsupported MODE in CONFIG. Use 'train' or 'predict'.")
