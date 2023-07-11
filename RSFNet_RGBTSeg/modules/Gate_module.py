from json.tool import main
from unicodedata import name
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1, with_relu=True):
        super(ConvBNReLU, self).__init__()
        self.with_relu = with_relu
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        out = self.bn(x)
        if self.with_relu:
            out = self.relu(out)
        return out

class GateFusion_EGCFM(nn.Module):
    def __init__(self, 
                 rgb_channels, 
                 inf_channels, 
                 mid_channels, 
                 out_channels, 
                 kernel_size=3,
                 op='concate', 
                 use_rep=True,
                 with_gate=False, 
                 with_skip=False,
                 early_skip=False):
        super(GateFusion_EGCFM, self).__init__()
        self.op = op
        self.use_rep = use_rep
        self.with_gate = with_gate
        self.with_skip = with_skip
        self.early_skip = early_skip

        padding = (kernel_size - 1) // 2
     
        self.rgb_1x1 = ConvBNReLU(rgb_channels, mid_channels, 1, 1, 0)
        if self.use_rep:
            from modules.egcfm import EnhancedGatedCrossFusionBlock
            self.rgb_kxk = EnhancedGatedCrossFusionBlock(mid_channels, mid_channels, kernel_size, 1, padding)
        else:
            self.rgb_kxk = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, padding, bias=False)
        self.rgb_fusion = ConvBNReLU(2 * mid_channels, out_channels, 1, 1, 0)
        # self.rgb_fusion = ConvBNReLU(mid_channels + rgb_channels, out_channels, 1, 1, 0)
      
        self.inf_1x1 = ConvBNReLU(inf_channels, mid_channels, 1, 1, 0)
        if self.use_rep:
            self.inf_kxk = EnhancedGatedCrossFusionBlock(mid_channels, mid_channels, kernel_size, 1, padding)
        else:
            self.inf_kxk = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, padding, bias=False)
        self.inf_fusion = ConvBNReLU( 2 * mid_channels, out_channels, 1, 1, 0)

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, rgb, inf, gate1, gate2):
        rgb_11 = self.rgb_1x1(rgb)
        inf_11 = self.inf_1x1(inf)
        rgb_33 = self.rgb_kxk(rgb_11)
        inf_33 = self.inf_kxk(inf_11)

        # cross fusion with other modal
        rgb_i = rgb_33 * self.sigmoid(inf_33)
        inf_r = inf_33 * self.sigmoid(rgb_33)

        # concate the feature
        rgb_c = torch.cat([rgb_i, rgb_11], dim=1)
        inf_c = torch.cat([inf_r, inf_11], dim=1)

        # fusion the cross feature and original feature
        rgb_f = self.rgb_fusion(rgb_c)
        inf_f = self.inf_fusion(inf_c)

        if self.with_skip and self.early_skip:
            rgb_f = rgb_f + rgb
            inf_f = inf_f + inf

        if self.with_gate:
            rgb_f = rgb_f * gate1
            inf_f = inf_f * gate2
        
        if self.with_skip and not self.early_skip:
            rgb_f = rgb_f + rgb
            inf_f = inf_f + inf

        if self.op == 'add':
            return rgb_f + inf_f
        elif self.op == 'concate':
            return torch.cat([rgb_f, inf_f], dim=1)
        else:
            return rgb_f, inf_f

if __name__ == '__main__':
    batches = 1
    gpu=1
    channel = 128
    size = 180
    rgb_feature = torch.randn(batches, channel, size, size).cuda(gpu)
    inf_feature = torch.randn(batches, channel, size, size).cuda(gpu)
    gate_fusion = GateFusion_EGCFM(channel, channel, 64, channel, op=None, use_rep=True).cuda(gpu)
    rgb_f, inf_f = gate_fusion(rgb_feature, inf_feature, 0, 0)
    print(rgb_f.size())
    from thop import profile
    total_ops, total_params  = profile(gate_fusion, inputs=(rgb_feature, inf_feature, 0, 0, ), verbose=False)
    print("%s | %s" % ("Params(K)", "FLOPs(M)"))
    print("%.2f | %.2f" % (total_params / 1000, total_ops / (1000 ** 2)))

    