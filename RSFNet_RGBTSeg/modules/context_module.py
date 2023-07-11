import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

# https://github.com/BangguWu/ECANet
class ECA_Compress(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channel, out_channel, k_size=3, auto_k=False, add=False):
        super(ECA_Compress, self).__init__()
        self.add=add
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if auto_k:
            gamma=2
            b=1
            t = int(abs((math.log(in_channel, 2) + b) / gamma))
            k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.compress = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        out = x * y.expand_as(x)

        if self.add:
            out += x

        out = self.compress(out)

        return out