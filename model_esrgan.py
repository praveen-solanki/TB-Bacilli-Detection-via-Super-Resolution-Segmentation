import torch
import torch.nn as nn
import math

# Residual in Residual Dense Block (RRDB) components
# A more accurate RRDB implementation
class DenseLayer(nn.Module):
    def __init__(self, in_feat, growth_rate=32):
        super().__init__()
        self.conv = nn.Conv2d(in_feat, growth_rate, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        out = self.lrelu(self.conv(x))
        return torch.cat((x, out), 1)

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        # Three dense layers chained together
        self.layer1 = DenseLayer(nf, gc)
        self.layer2 = DenseLayer(nf + gc, gc)
        self.layer3 = DenseLayer(nf + 2 * gc, gc)
        # A final 1x1 conv to fuse the features
        self.conv_1x1 = nn.Conv2d(nf + 3 * gc, nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.conv_1x1(out))
        # Residual scaling (beta) and outer residual connection
        return out * 0.2 + identity
    
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_nc, nf, 3,1,1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3,1,1)
        # upsample blocks (x2 x2 for scale 4)
        up_layers = []
        n_upscale = int(math.log(scale,2))
        for _ in range(n_upscale):
            up_layers += [nn.Conv2d(nf, nf*4, 3,1,1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True)]
        self.upsampler = nn.Sequential(*up_layers)
        self.conv_last = nn.Conv2d(nf, out_nc, 3,1,1)

    def forward(self, x):
        fea = self.conv_first(x)
        body = self.body(fea)
        fea = fea + self.conv_body(body)
        out = self.upsampler(fea)
        out = self.conv_last(out)
        return out

# Simple Patch discriminator
class Discriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super().__init__()
        def block(in_ch, out_ch, stride=1):
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride, 1), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, True)]
            return layers
        layers=[]
        layers += block(in_nc, nf, 1)
        layers += block(nf, nf*2, 2)
        layers += block(nf*2, nf*4, 2)
        layers += block(nf*4, nf*8, 2)
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(nf*8,1)]
        self.model = nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)
