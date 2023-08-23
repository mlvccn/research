import torch
import torch.nn as nn
import torch.nn.functional as F

class PyrUpBicubic2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        def kernel(d):
            x = d + torch.arange(-1, 3, dtype=torch.float32)
            x = torch.abs(x)
            a = -0.75
            f = (x < 1).float() * ((a + 2) * x * x * x - (a + 3) * x * x + 1) + \
                ((x >= 1) * (x < 2)).float() * (a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a)
            W = f.reshape(1, 1, 1, len(x)).float()
            Wt = W.permute(0, 1, 3, 2)
            return W, Wt

        We, We_t = kernel(-0.25)
        Wo, Wo_t = kernel(-0.25 - 0.5)

        # Building non-separable filters for now. It would make sense to
        # have separable filters if it proves to be faster.

        # .contiguous() is needed until a bug is fixed in nn.Conv2d.
        self.W00 = (We_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W01 = (We_t @ Wo).expand(channels, 1, 4, 4).contiguous()
        self.W10 = (Wo_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W11 = (Wo_t @ Wo).expand(channels, 1, 4, 4).contiguous()

    def forward(self, input):

        if input.device != self.W00.device:
            self.W00 = self.W00.to(input.device)
            self.W01 = self.W01.to(input.device)
            self.W10 = self.W10.to(input.device)
            self.W11 = self.W11.to(input.device)

        a = F.pad(input, (2, 2, 2, 2), 'replicate')

        I00 = F.conv2d(a, self.W00, groups=self.channels)
        I01 = F.conv2d(a, self.W01, groups=self.channels)
        I10 = F.conv2d(a, self.W10, groups=self.channels)
        I11 = F.conv2d(a, self.W11, groups=self.channels)

        n, c, h, w = I11.shape

        J0 = torch.stack((I00, I01), dim=-1).view(n, c, h, 2 * w)
        J1 = torch.stack((I10, I11), dim=-1).view(n, c, h, 2 * w)
        out = torch.stack((J0, J1), dim=-2).view(n, c, 2 * h, 2 * w)

        out = F.pad(out, (-1, -1, -1, -1))
        return out


class BackwardCompatibleUpsampler(nn.Module):
    """ Upsampler with bicubic interpolation that works with Pytorch 1.0.1 """

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.up1 = PyrUpBicubic2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 3, padding=1)
        self.up2 = PyrUpBicubic2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, image_size):
        x = self.up1(x)
        x = self.relu(self.conv1(x))
        x = self.up2(x)
        x = F.interpolate(x, image_size[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x