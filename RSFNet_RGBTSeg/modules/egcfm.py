import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.dbb_transforms import *

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Conv2d):

    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).type_as(self.weight)
        nn.init.zeros_(self.weight)

    def forward(self, input):
        kernel = self.weight + self.id_tensor.to(self.weight.device)
        result = F.conv2d(input, kernel, None, stride=1, padding=0, dilation=self.dilation, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor.to(self.weight.device)



class BNAndPadLayer(nn.Module):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = self.bn.bias.detach() - self.bn.running_mean * self.bn.weight.detach() / torch.sqrt(self.bn.running_var + self.bn.eps)
            else:
                pad_values = - self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0:self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0:self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class EnhancedGatedCrossFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, deploy=False, nonlinear=None):
        super(EnhancedGatedCrossFusionBlock, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        assert padding == kernel_size // 2

        if deploy:
            self.dbb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

        else:
            # original kxk convolution
            self.dbb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, \
                kernel_size=kernel_size, stride=stride, padding=padding)
            # 1x1 convolution extend
            self.dbb_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, \
                kernel_size=1, stride=stride, padding=0)
            hor_padding = (0, padding) # (1, kernel_size)
            ver_padding = (padding, 0) # (kernel_size, 1)
            self.dbb_1x1_1xk = nn.Sequential()
            self.dbb_1x1_1xk.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                            kernel_size=1, stride=1, padding=0, bias=False))
            self.dbb_1x1_1xk.add_module('conv1xk', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
              kernel_size=(1, kernel_size), stride=stride, padding=hor_padding, bias=False))
         
            self.dbb_1x1_kx1 = nn.Sequential()
            self.dbb_1x1_kx1.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                            kernel_size=1, stride=1, padding=0, bias=False))
            self.dbb_1x1_kx1.add_module('convkx1', nn.Conv2d(in_channels=in_channels , out_channels=out_channels, \
              kernel_size=(kernel_size, 1), stride=stride, padding=ver_padding, bias=False))


    def forward(self, inputs):
        if hasattr(self, 'dbb_reparam'):
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        out += self.dbb_1x1(inputs)
        out += self.dbb_1x1_1xk(inputs)
        out += self.dbb_1x1_kx1(inputs)
        return self.nonlinear(out)

    

if __name__ == '__main__':
    x = torch.randn(1, 64, 56, 56)
    k=7
    s=1
    model = EnhancedGatedCrossFusionBlock(in_channels=64, out_channels=64, kernel_size=k, \
        stride=s, padding=k//2, deploy=False)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)
    model.eval()
    print(model)
    train_y = model(x)
    print(train_y.size())