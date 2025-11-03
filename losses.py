import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class VGGFeatureLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        # use layers up to relu_5_4 maybe up to conv4_4
        self.features = nn.Sequential(*list(vgg.children())[:20]).eval()
        for p in self.features.parameters():
            p.requires_grad=False
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # expects 0-1 scaled, convert to normalized for VGG
        # VGG expects 0-1 then normalized with mean/std
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(x.device)
        std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(x.device)
        x_n = (x - mean)/std
        y_n = (y - mean)/std
        fx = self.features(x_n)
        fy = self.features(y_n)
        return self.criterion(fx, fy)

l1_loss = nn.L1Loss()
bce_loss = nn.BCEWithLogitsLoss()
