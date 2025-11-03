import torch
from PIL import Image, ImageDraw, ImageFont
import os
from model_esrgan import RRDBNet
import torchvision.transforms.functional as TF
import numpy as np
from torchvision.utils import save_image
import cv2
import torch.nn.functional as F
from dataset import degrade_image

def load_hr_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return img[:,:,::-1]

def save_side_by_side_pil(imgs, labels, out_path, label_height=28, pad=8, bg=(0,0,0)):
    assert len(imgs) == len(labels)
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]
    max_h = max(heights)
    total_w = sum(widths) + pad * (len(imgs)-1)
    canvas_h = max_h + label_height + pad
    canvas = Image.new("RGB", (total_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for im, lab, w, h in zip(imgs, labels, widths, heights):
        y_img = (max_h - h) // 2
        canvas.paste(im, (x, y_img))
        text_w, text_h = draw.textsize(lab, font=font)
        text_x = x + (w - text_w)//2
        text_y = max_h + ((label_height - text_h)//2)
        draw.text((text_x, text_y), lab, fill=(255,255,255), font=font)
        x += w + pad
    canvas.save(out_path)

def run_inference(ckpt, input_folder, output_folder, scale=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = RRDBNet(nf=64, nb=8, scale=scale).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()
    os.makedirs(output_folder, exist_ok=True)

    all_files = os.listdir(input_folder)
    print(f"Found {len(all_files)} files in the input folder.")

    for f in all_files:
        try:
            valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
            if not f.lower().endswith(valid_extensions):
                print(f"--> Skipping file: {f} (invalid extension)")
                continue

            print(f"--> Processing file: {f}")
            image_path = os.path.join(input_folder, f)
            hr_original_np = load_hr_image(image_path)

            if hr_original_np is None:
                print(f"!!!!!! ERROR: Failed to load {f}. Skipping. !!!!!!")
                continue

            lr_np, _ = degrade_image(hr_original_np, scale=scale)

            lr_tensor = torch.from_numpy(lr_np.transpose(2,0,1)).float().div(255.0).unsqueeze(0).to(device)
            hr_original_tensor = torch.from_numpy(hr_original_np.transpose(2,0,1)).float().div(255.0).unsqueeze(0)

            with torch.no_grad():
                sr_tensor = G(lr_tensor)
                sr_tensor = sr_tensor.clamp(0,1)

            target_h, target_w = hr_original_tensor.shape[2:]
            lr_upsampled_tensor = F.interpolate(lr_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # convert to PIL
            lr_pil = TF.to_pil_image(lr_upsampled_tensor.squeeze(0).cpu())
            sr_pil = TF.to_pil_image(sr_tensor.squeeze(0).cpu())
            hr_pil = TF.to_pil_image(hr_original_tensor.squeeze(0))

            output_filename = os.path.splitext(f)[0] + "_comparison.png"
            save_side_by_side_pil([lr_pil, sr_pil, hr_pil],
                                  ["LR (upscaled)", "SR (model)", "HR (original)"],
                                  os.path.join(output_folder, output_filename))
            print(f"    ✅ Successfully saved: {output_filename}")

        except Exception as e:
            print(f"!!!!!! An unexpected error occurred while processing {f}: {e} !!!!!!")
            continue
