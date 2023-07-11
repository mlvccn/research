from bz2 import compress
import torch
from torch import nn
import torch.nn.functional as F

from model.unet10 import ResUnet
from modules.context_module import ECA_Compress
from modules.Gate_module import GateFusion_EGCFM


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


""" fusion two level features """
class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channels, norm_layer=nn.BatchNorm2d):
        super(FAM, self).__init__()
        self.conv_d21 = nn.Conv2d(in_channel_down, in_channel_left, kernel_size=3, stride=1, padding=1)
        self.conv_l2d = nn.Conv2d(in_channel_left, in_channel_down, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channel_left + in_channel_down, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channels)

    def forward(self, left, down):
        down_mask = self.conv_d21(down)
        left_mask = self.conv_l2d(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=True)
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear', align_corners=True)

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, n_class):
        super(SegHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(mid_channels, n_class, kernel_size=1)
            )

    def forward(self, x):
        return self.cls(x)


class DPAdd50_EGCFM_k5_addHead(nn.Module):
    def __init__(self, layers=18, n_class=9, pretrained=True, with_gate=True, with_skip=True, early_skip=False):
        super(DPAdd50_EGCFM_k5_addHead, self).__init__()
        rgb_in = [256, 512, 1024, 2048]
        if layers == 18 or layers == 34:
            thermal_in = [64, 128, 256, 512]
        else:
            thermal_in = rgb_in
        rgb_out = [64, 128, 256, 256]
        thermal_out = rgb_out
        self.conv1x1 = False
        self.compress = True
        self.early_fusion = True
        # UNet for RGB 
        unet_rgb = ResUnet(layers=50, out_dims=rgb_out, classes=n_class)
        self.layer0_rgb = unet_rgb.layer0
        self.layer1_rgb = unet_rgb.layer1
        self.layer2_rgb = unet_rgb.layer2
        self.layer3_rgb = unet_rgb.layer3
        self.layer4_rgb = unet_rgb.layer4
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(rgb_in[3], 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
      
        # UNer for Thermal
        unet_th = ResUnet(layers=layers, out_dims=thermal_out, classes=n_class, pretrained=pretrained)
        self.layer0_th = unet_th.layer0
        self.layer1_th = unet_th.layer1
        self.layer2_th = unet_th.layer2
        self.layer3_th = unet_th.layer3
        self.layer4_th = unet_th.layer4
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.fc2 = nn.Sequential(
            nn.Linear(thermal_in[3], 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


        if self.compress:
            if self.conv1x1:
                self.rgb_down4 = nn.Conv2d(rgb_in[3], rgb_out[3], 1, 1, 0)
                self.rgb_down3 = nn.Conv2d(rgb_in[2], rgb_out[2], 1, 1, 0)
                self.rgb_down2 = nn.Conv2d(rgb_in[1], rgb_out[1], 1, 1, 0)
                self.rgb_down1 = nn.Conv2d(rgb_in[0], rgb_out[0], 1, 1, 0)

                self.th_down4 = nn.Conv2d(thermal_in[3], thermal_out[3], 1, 1, 0)
                self.th_down3 = nn.Conv2d(thermal_in[2], thermal_out[2], 1, 1, 0)
                self.th_down2 = nn.Conv2d(thermal_in[1], thermal_out[1], 1, 1, 0)
                self.th_down1 = nn.Conv2d(thermal_in[0], thermal_out[0], 1, 1, 0)
            else:
                self.rgb_down4 = ECA_Compress(rgb_in[3], rgb_out[3], auto_k=True, add=True)
                self.rgb_down3 = ECA_Compress(rgb_in[2], rgb_out[2], auto_k=True, add=True)
                self.rgb_down2 = ECA_Compress(rgb_in[1], rgb_out[1], auto_k=True, add=True)
                self.rgb_down1 = ECA_Compress(rgb_in[0], rgb_out[0], auto_k=True, add=True)

                self.th_down4 = ECA_Compress(thermal_in[3], thermal_out[3], auto_k=True, add=True)
                self.th_down3 = ECA_Compress(thermal_in[2], thermal_out[2], auto_k=True, add=True)
                self.th_down2 = ECA_Compress(thermal_in[1], thermal_out[1], auto_k=True, add=True)
                self.th_down1 = ECA_Compress(thermal_in[0], thermal_out[0], auto_k=True, add=True)
        
        else:
            rgb_out = rgb_in
            thermal_out = thermal_in

        kernel_size=5; op=None; use_rep=False
        self.op=op
        mid_channel=64 # 32, 64(default), 128, 256
        print("mid_channel: ", mid_channel)
        self.gate4 = GateFusion_EGCFM(rgb_out[3], thermal_out[3], mid_channel, rgb_out[3], \
            kernel_size, op, use_rep, with_gate, with_skip, early_skip)
        self.gate3 = GateFusion_EGCFM(rgb_out[2], thermal_out[2], mid_channel, rgb_out[2], \
            kernel_size, op, use_rep, with_gate, with_skip, early_skip)
        self.gate2 = GateFusion_EGCFM(rgb_out[1], thermal_out[1], mid_channel, rgb_out[1], \
            kernel_size, op, use_rep, with_gate, with_skip, early_skip)
        self.gate1 = GateFusion_EGCFM(rgb_out[0], thermal_out[0], mid_channel, rgb_out[0], \
            kernel_size, op, use_rep, with_gate, with_skip, early_skip)
        self.fam54_rgb = FAM(rgb_out[2], rgb_out[3], rgb_out[2])
        self.fam43_rgb = FAM(rgb_out[1], rgb_out[2], rgb_out[1])
        self.fam32_rgb = FAM(rgb_out[0], rgb_out[1], rgb_out[0])

        # for infrared
        self.fam54_th = FAM(thermal_out[2], thermal_out[3], thermal_out[2])
        self.fam43_th = FAM(thermal_out[1], thermal_out[2], thermal_out[1])
        self.fam32_th = FAM(thermal_out[0], thermal_out[1], thermal_out[0])
        self.cls = SegHead(rgb_out[0], 128, n_class)
        self.aux1 = SegHead(rgb_out[1], 128, n_class)
        self.aux2 = SegHead(rgb_out[2], 128, n_class)


    def forward(self, x):
        bz = x.shape[0]
        rgb = x[:,:3]
        thermal = x[:,3:]
        thermal = thermal.repeat(1, 3, 1, 1)

        feat_rgb =  self.layer0_rgb(rgb)
        feat_rgb_4 = self.layer1_rgb(feat_rgb) 
        feat_rgb_8 = self.layer2_rgb(feat_rgb_4)
        feat_rgb_16 = self.layer3_rgb(feat_rgb_8)
        feat_rgb_32 = self.layer4_rgb(feat_rgb_16)
        gap_rgb = self.gap1(feat_rgb_32)
        gap_rgb = gap_rgb.view(bz, -1)
        gate_rgb = self.fc1(gap_rgb)

        feat_th =  self.layer0_th(thermal)
        feat_th_4 = self.layer1_th(feat_th)
        feat_th_8 = self.layer2_th(feat_th_4)
        feat_th_16 = self.layer3_th(feat_th_8)
        feat_th_32 = self.layer4_th(feat_th_16)
        gap_th = self.gap2(feat_th_32)
        gap_th = gap_th.view(bz, -1)
        gate_th = self.fc2(gap_th)

        gate_rgb = gate_rgb / (gate_rgb + gate_th + 1e-7)
        gate_th = 1.0 - gate_rgb
        gate_rgb = gate_rgb.view(bz, 1, 1, 1)
        gate_th = gate_th.view(bz, 1, 1, 1)

        # Compress the skip conneciton output channels
        if self.compress:
            feat_rgb_4 = self.rgb_down1(feat_rgb_4)
            feat_rgb_8 = self.rgb_down2(feat_rgb_8)
            feat_rgb_16 = self.rgb_down3(feat_rgb_16)
            feat_rgb_32 = self.rgb_down4(feat_rgb_32)

            feat_th_4 = self.th_down1(feat_th_4)
            feat_th_8 = self.th_down2(feat_th_8)
            feat_th_16 = self.th_down3(feat_th_16)
            feat_th_32 = self.th_down4(feat_th_32)

        if self.early_fusion:
            # early fusion

            if self.op is None:
                feat_rgb_fuse_32, feat_th_fuse_32 = self.gate4(feat_rgb_32, feat_th_32, gate_rgb, gate_th)
                feat_rgb_fuse_16, feat_th_fuse_16 = self.gate3(feat_rgb_16, feat_th_16, gate_rgb, gate_th)
                feat_rgb_fuse_8, feat_th_fuse_8 = self.gate2(feat_rgb_8, feat_th_8, gate_rgb, gate_th)
                feat_rgb_fuse_4, feat_th_fuse_4 = self.gate1(feat_rgb_4, feat_th_4, gate_rgb, gate_th)

                # 1/16
                out_rgb_16 = self.fam54_rgb(feat_rgb_fuse_16, feat_rgb_fuse_32)
                out_th_16 = self.fam54_th(feat_th_fuse_16, feat_th_fuse_32)

                # 1/8
                out_rgb_8 = self.fam43_rgb(feat_rgb_fuse_8, out_rgb_16)
                out_th_8 = self.fam43_th(feat_th_fuse_8, out_th_16)

                # 1/4
                out_rgb_4 = self.fam32_rgb(feat_rgb_fuse_4, out_rgb_8)
                out_th_4 = self.fam32_th(feat_th_fuse_4, out_th_8)
            else:
                feat_fusion_32 = self.gate4(feat_rgb_32, feat_th_32, gate_rgb, gate_th)
                feat_fusion_16 = self.gate3(feat_rgb_16, feat_th_16, gate_rgb, gate_th)
                feat_fusion_8 = self.gate2(feat_rgb_8, feat_th_8, gate_rgb, gate_th)
                feat_fusion_4 = self.gate1(feat_rgb_4, feat_th_4, gate_rgb, gate_th)

                # 1/16
                out_rgb_16 = self.fam54_rgb(feat_fusion_16, feat_fusion_32)
                out_th_16 = self.fam54_th(feat_fusion_16, feat_fusion_32)

                # 1/8
                out_rgb_8 = self.fam43_rgb(feat_fusion_8, out_rgb_16)
                out_th_8 = self.fam43_th(feat_fusion_8, out_th_16)

                # 1/4
                out_rgb_4 = self.fam32_rgb(feat_fusion_4, out_rgb_8)
                out_th_4 = self.fam32_th(feat_fusion_4, out_th_8)

            out = self.cls(out_rgb_4 + out_th_4)
            out = F.interpolate(out, scale_factor=4, \
                mode='bilinear', align_corners=True)

            if self.training:
                out_8 = self.aux1(out_rgb_8 + out_th_8)
                out_8 = F.interpolate(out_8, scale_factor=8, \
                    mode='bilinear', align_corners=True)

                out_16 = self.aux2(out_rgb_16 + out_th_16)
                out_16 = F.interpolate(out_16, scale_factor=16, \
                    mode='bilinear', align_corners=True)
                return out, gate_rgb.view(bz, -1), gate_th.view(bz, -1), \
                    out_8, out_16
            else:
                return out


if __name__ == '__main__':
    num_minibatch = 1
    gpu=1
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(gpu)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(gpu)
    model = DPAdd50_EGCFM_k5_addHead(layers=18, n_class=9).cuda(gpu)
    model.eval()
    input = torch.cat((rgb, thermal), dim=1)
    out = model(input)
    print(model)
    print("The output size: {}".format(out[0].size()))

    from thop import profile
    total_ops, total_params  = profile(model, inputs=(input, ), verbose=False)
    print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    print("%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
