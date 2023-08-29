from torch import nn
import torch

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)
        self.ffc1 = FFCResnetBlock(C_hid,C_out)
        # self.ffc2 = FFCResnetBlock(C_hid,C_out)
        # self.ffc3 = FFCResnetBlock(C_hid,C_out)
    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
            y += self.ffc1(x)
            # y += self.ffc2(x)
            # y += self.ffc3(x)
        return y


#############################  inpainting part ######################
class FourierUnit(nn.Module):

    
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)#nn.GroupNorm(groups,out_channels * 2)
        self.relu = nn.ReLU(inplace=True)#nn.LeakyReLU(0.2, inplace=True)
        # self.bn = nn.GroupNorm(groups,out_channels * 2)
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        batch, c, h, w = x.size()   #[B,CxT,H,W]
        fft_dim = (-2,-1)
        ffted = torch.fft.rfftn(x,dim=fft_dim,norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        # (batch, c, h, w/2+1, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        # (batch, c, 2, h, w/2+1)
        
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[...,0],ffted[...,1])
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho') 

        return output  #(batch,c*t,h,w)


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            # nn.GroupNorm(groups,out_channels // 2),
            # nn.LeakyReLU(0.2,inplace=True)
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        # ratio = 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            #out_xg = self.convl2g(x_l) + self.convg2g(x_g)
            out_xg =  self.convg2g(x_g)
        
        return out_xl, out_xg


class FFCResnetBlock(nn.Module):
    def __init__(self, in_dim,out_dim,  norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dilation=1,
                  inline=True, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(in_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        
        id_l, id_g = x_l, x_g
        
        x_l, x_g = self.conv1((x_l, x_g))
        # if x_l.size(1)== id_l.size(1):
        #    x_l, x_g = id_l + x_l, id_g + x_g
        # out = x_l, x_g
        # if self.inline:
        #   out = torch.cat(out, dim=1)
        # return out
        return x_g


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin=1, ratio_gout=1,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer #nn.GroupNorm 
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))
        
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer# nn.LeakyReLU #
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)
        
    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g
#####################################################################

# class FourierUnit(nn.Module):

#     def __init__(self, in_channels, out_channels, groups=1):
#         # bn_layer not used
#         super(FourierUnit, self).__init__()
#         self.groups = groups
#         self.conv_layer = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
#                                           kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels * 2)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         batch, c, h, w = x.size()   #[B,CxT,H,W]
#         fft_dim = (-2,-1)
#         ffted = torch.fft.rfftn(x,dim=fft_dim,norm='ortho')
#         ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
#         # (batch, c, h, w/2+1, 2)
#         ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
#         ffted = ffted.view((batch, -1,) + ffted.size()[3:])
#         # (batch, c, 2, h, w/2+1)

#         ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
#         ffted = self.relu(self.bn(ffted))

#         ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
#             0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
#         ffted = torch.complex(ffted[...,0],ffted[...,1])
#         ifft_shape_slice = x.shape[-2:]
#         output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho') 

#         return output  #(batch,c*t,h,w)


# class SpectralTransform(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
#         # bn_layer not used
#         super(SpectralTransform, self).__init__()
#         self.enable_lfu = enable_lfu
#         if stride == 2:
#             self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         else:
#             self.downsample = nn.Identity()

#         self.stride = stride
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels //
#                       2, kernel_size=1, groups=groups, bias=False),
#             nn.BatchNorm2d(out_channels // 2),
#             nn.ReLU(inplace=True)
#         )
#         self.fu = FourierUnit(
#             out_channels // 2, out_channels // 2, groups)
#         if self.enable_lfu:
#             self.lfu = FourierUnit(
#                 out_channels // 2, out_channels // 2, groups)
#         self.conv2 = nn.Conv2d(
#             out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

#     def forward(self, x):

#         x = self.downsample(x)
#         x = self.conv1(x)
#         output = self.fu(x)

#         if self.enable_lfu:
#             n, c, h, w = x.shape
#             split_no = 2
#             split_s_h = h // split_no
#             split_s_w = w // split_no
#             xs = torch.cat(torch.split(
#                 x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
#             xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
#                            dim=1).contiguous()
#             xs = self.lfu(xs)
#             xs = xs.repeat(1, 1, split_no, split_no).contiguous()
#         else:
#             xs = 0

#         output = self.conv2(x + output + xs)

#         return output


# class FFC(nn.Module):

#     def __init__(self, in_channels, out_channels, kernel_size,
#                  ratio_gin, ratio_gout, stride=1, padding=0,
#                  dilation=1, groups=1, bias=False, enable_lfu=True):
#         super(FFC, self).__init__()

#         assert stride == 1 or stride == 2, "Stride should be 1 or 2."
#         self.stride = stride

#         in_cg = int(in_channels * ratio_gin)
#         in_cl = in_channels - in_cg
#         out_cg = int(out_channels * ratio_gout)
#         out_cl = out_channels - out_cg
#         #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
#         #groups_l = 1 if groups == 1 else groups - groups_g
        
#         self.ratio_gin = ratio_gin
#         self.ratio_gout = ratio_gout
#         self.global_in_num = in_cg

#         module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
#         self.convl2l = module(in_cl, out_cl, kernel_size,
#                               stride, padding, dilation, groups, bias)
#         module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
#         self.convl2g = module(in_cl, out_cg, kernel_size,
#                               stride, padding, dilation, groups, bias)
#         module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
#         self.convg2l = module(in_cg, out_cl, kernel_size,
#                               stride, padding, dilation, groups, bias)
#         module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
#         self.convg2g = module(
#             in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

#     def forward(self, x):
#         x_l, x_g = x if type(x) is tuple else (x, 0)
#         out_xl, out_xg = 0, 0

#         if self.ratio_gout != 1:
#             out_xl = self.convl2l(x_l) + self.convg2l(x_g)
#         if self.ratio_gout != 0:
#             out_xg = self.convl2g(x_l) + self.convg2g(x_g)

#         return out_xl, out_xg

# class FFC_BN_ACT(nn.Module):

#     def __init__(self, in_channels, out_channels,
#                  kernel_size, ratio_gin=0.5, ratio_gout=0.5,
#                  stride=1, padding=0, dilation=1, groups=1, bias=False,
#                  norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
#                  enable_lfu=True):
#         super(FFC_BN_ACT, self).__init__()
#         self.ffc = FFC(in_channels, out_channels, kernel_size,
#                        ratio_gin, ratio_gout, stride, padding, dilation,
#                        groups, bias, enable_lfu)
#         lnorm = nn.Identity if ratio_gout == 1 else norm_layer
#         gnorm = nn.Identity if ratio_gout == 0 else norm_layer
#         self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
#         self.bn_g = gnorm(int(out_channels * ratio_gout))

#         lact = nn.Identity if ratio_gout == 1 else activation_layer
#         gact = nn.Identity if ratio_gout == 0 else activation_layer
#         self.act_l = lact(inplace=True)
#         self.act_g = gact(inplace=True)

#     def forward(self, x):
#         x_l, x_g = self.ffc(x)
#         x_l = self.act_l(self.bn_l(x_l))
#         x_g = self.act_g(self.bn_g(x_g))
#         return x_l, x_g