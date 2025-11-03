import os
import random
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2

def degrade_image(hr, scale=4):
    img = hr.copy()

    # 1. Random Gaussian blur (50% chance)
    if random.random() < 0.5:
        sigma = random.uniform(1, 8)
        img = cv2.GaussianBlur(img, (0,0), sigmaX=sigma)

    h, w, _ = img.shape
    new_h, new_w = h // scale, w // scale

    # 2. Downsample (random interpolation method)
    lr = cv2.resize(img, (new_w, new_h),
                    interpolation=random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR]))

    # 3. JPEG compression artifacts (40% chance)
    if random.random() < 0.4:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(40, 90)]
        _, encimg = cv2.imencode('.jpg', lr, encode_param)
        lr = cv2.imdecode(encimg, 1)

    # 4. Add Gaussian noise (50% chance)
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(2, 10), lr.shape).astype(np.float32)
        lr = np.clip(lr + noise, 0, 255).astype(np.uint8)

    return lr, hr

def save_side_by_side(images, labels, save_path):
    from PIL import Image, ImageDraw, ImageFont

    # Canvas: width = sum of widths, height = max height
    widths, heights = zip(*(im.size for im in images))
    total_width, max_height = sum(widths), max(heights)
    new_im = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # Paste images side by side
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Draw labels
    draw = ImageDraw.Draw(new_im)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    x_offset = 0
    for im, lab in zip(images, labels):
        bbox = draw.textbbox((0, 0), lab, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x_offset + (im.size[0] - text_w) // 2
        y = max_height - text_h - 10
        draw.text((x, y), lab, font=font, fill=(0, 0, 0))
        x_offset += im.size[0]

    new_im.save(save_path)

class PairedMicroscopyDataset(Dataset):
    def __init__(self, hr_folder, patch_size=256, scale=4, transforms=None):
        self.files = [os.path.join(hr_folder,f) for f in os.listdir(hr_folder) if f.lower().endswith(('.png','.jpg','.tif','.tiff'))]
        self.scale=scale
        self.transforms=transforms
        self.patch_size = max((patch_size // scale) * scale, 128)  # minimum 128x128
        self._saved_pairs = set()
        os.makedirs("pairs/side_by_side", exist_ok=True)

    def __len__(self):
        return len(self.files)

    def random_crop(self, img, size):
        h,w,_ = img.shape
        if h < size or w < size:
            pad_h = max(size-h,0)
            pad_w = max(size-w,0)
            img = cv2.copyMakeBorder(img,0,pad_h,0,pad_w, cv2.BORDER_REFLECT)
            h,w,_ = img.shape
        top = random.randint(0, h-size)
        left = random.randint(0, w-size)
        return img[top:top+size, left:left+size]

    def __getitem__(self, idx):
        hr_path = self.files[idx]
        img_name = os.path.splitext(os.path.basename(hr_path))[0]

        # ---- Load HR image ----
        hr = cv2.imread(hr_path)[:,:,::-1]  # BGR->RGB
        hr_patch = self.random_crop(hr, self.patch_size)

        # ---- Generate degraded LR ----
        lr_patch, hr_patch = degrade_image(hr_patch, scale=self.scale)

        # ---- Ensure contiguous ----
        hr_patch = hr_patch.copy()
        lr_patch = lr_patch.copy()

        # ---- Save [LR | LR_upscaled | HR] for inspection (once per filename) ----
        out_name = f"pairs/side_by_side/{img_name}_comparison.png"
        if img_name not in self._saved_pairs:
            # create PIL images
            lr_pil = Image.fromarray(lr_patch)
            # upscale LR to HR size for visualization
            lr_up = cv2.resize(lr_patch, (hr_patch.shape[1], hr_patch.shape[0]), interpolation=cv2.INTER_CUBIC)
            lr_up_pil = Image.fromarray(lr_up)
            hr_pil = Image.fromarray(hr_patch)
            save_side_by_side([lr_pil, lr_up_pil, hr_pil],
                              ["LR (degraded)", "LR upscaled", "HR (original)"],
                              out_name)
            self._saved_pairs.add(img_name)

        # ---- Convert to torch tensors ----
        hr_t = torch.from_numpy(hr_patch.transpose(2,0,1)).float().div(255.0)
        lr_t = torch.from_numpy(lr_patch.transpose(2,0,1)).float().div(255.0)

        return lr_t, hr_t
