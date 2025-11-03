import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import PairedMicroscopyDataset
from model_esrgan import RRDBNet, Discriminator
from losses import VGGFeatureLoss, l1_loss, bce_loss
import torch.optim as optim
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np
import torch.nn.functional as F

def calc_metrics(pred, gt):
    p = pred.detach().cpu().numpy().transpose(1,2,0)
    g = gt.detach().cpu().numpy().transpose(1,2,0)
    p = np.clip(p, 0, 1)
    g = np.clip(g, 0, 1)
    psnr_val = psnr_metric(g, p, data_range=1.0)
    h, w, _ = g.shape
    win_size = 7
    if min(h, w) < 7:
        win_size = (min(h, w) // 2) * 2 + 1
    ssim_val = ssim_metric(
        (g*255).astype(np.uint8),
        (p*255).astype(np.uint8),
        channel_axis=2,
        win_size=win_size
    )
    return psnr_val, ssim_val

def save_side_by_side_pil(images, labels, save_path):
    from PIL import Image, ImageDraw, ImageFont

    # Canvas size
    widths, heights = zip(*(im.size for im in images))
    total_width, max_height = sum(widths), max(heights)
    new_im = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    # Paste images
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
        # ✅ Pillow 10+ replacement for textsize()
        bbox = draw.textbbox((0, 0), lab, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = x_offset + (im.size[0] - text_w) // 2
        y = max_height - text_h - 10
        draw.text((x, y), lab, font=font, fill=(0, 0, 0))
        x_offset += im.size[0]

    new_im.save(save_path)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedMicroscopyDataset(args.hr_folder, patch_size=args.patch_size, scale=args.scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    G = RRDBNet(nf=64, nb=8, scale=args.scale).to(device)
    if args.adv:
        D = Discriminator().to(device)
        optD = optim.Adam(D.parameters(), lr=args.lr*0.1, betas=(0.9,0.999))
    else:
        D = None

    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.9,0.999))
    vgg = VGGFeatureLoss(device=device).to(device)
    step=0

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(loader), total=len(loader))
        for i,(lr,hr) in pbar:
            lr = lr.to(device)
            hr = hr.to(device)
            # generator forward
            sr = G(lr)
            sr = sr[:, :, :hr.shape[2], :hr.shape[3]]  # crop to match HR
            loss_pixel = l1_loss(sr, hr)
            loss_vgg = vgg(sr, hr) * args.lambda_vgg
            loss_g = loss_pixel + loss_vgg

            if args.adv:
                D.train()
                optD.zero_grad()
                real_out = D(hr).view(-1)
                fake_out = D(sr.detach()).view(-1)
                loss_d = (bce_loss(real_out, torch.ones_like(real_out)) + bce_loss(fake_out, torch.zeros_like(fake_out))) * 0.5
                loss_d.backward()
                optD.step()
                fake_out_for_g = D(sr).view(-1)
                loss_adv_g = bce_loss(fake_out_for_g, torch.ones_like(fake_out_for_g)) * args.lambda_adv
                loss_g = loss_g + loss_adv_g

            optG.zero_grad()
            loss_g.backward()
            optG.step()

            if step % args.log_interval == 0:
                psnr_val, ssim_val = calc_metrics(sr[0].clamp(0,1), hr[0].clamp(0,1))
                pbar.set_description(f"e{epoch} it{step} Lpix:{loss_pixel.item():.4f} Lvgg:{loss_vgg.item():.4f} PSNR:{psnr_val:.2f}")

            if step % args.save_interval == 0:
                save_path = os.path.join(args.ckpt_dir, f"g_step{step}.pth")
                torch.save(G.state_dict(), save_path)
                if D is not None:
                    torch.save(D.state_dict(), os.path.join(args.ckpt_dir, f"d_step{step}.pth"))

                # prepare visualization images (PIL)
                # LR upsampled to HR for visualization
                lr_vis = F.interpolate(lr[0].unsqueeze(0), size=hr[0].shape[1:], mode='bilinear', align_corners=False)
                lr_vis = lr_vis[0].cpu()
                sr_vis = sr[0].clamp(0,1).cpu()
                hr_vis = hr[0].cpu()

                # convert to PIL images
                lr_pil = TF.to_pil_image(lr_vis)
                sr_pil = TF.to_pil_image(sr_vis)
                hr_pil = TF.to_pil_image(hr_vis)

                out_file = os.path.join(args.outputs, f"sample_{step}.png")
                save_side_by_side_pil([lr_pil, sr_pil, hr_pil],
                                      ["LR (up)", "SR (model)", "HR (orig)"],
                                      out_file)
            step+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr_folder', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adv', action='store_true', help='enable adversarial training')
    parser.add_argument('--lambda_vgg', type=float, default=0.01)
    parser.add_argument('--lambda_adv', type=float, default=0.001)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=500)
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.outputs, exist_ok=True)
    train(args)
