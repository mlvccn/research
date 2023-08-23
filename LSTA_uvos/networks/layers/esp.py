import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.activation import AconC

from torch.nn import SyncBatchNorm as norm


@torch.jit.script
def swish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.sigmoid(input)


class Swish(nn.Module):
    '''
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return swish(input)

def relu(negative_slope=0.0, inplace=False):
    # return nn.LeakyReLU(negative_slope, inplace=inplace)
    # relu =
    return Swish()

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        #self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        #self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = norm(nOut, eps=1e-03)
        self.act = relu()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        #output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = norm(nOut, eps=1e-03)
        self.act = relu()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = norm(nOut, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class Dilatedconvolution(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        if isinstance(kSize, tuple) and len(kSize)==2:
            k1,k2 = kSize
        elif isinstance(kSize, int):
            k1 = k2 = kSize
        else:
            raise NotImplementedError
        padding1 = int((k1 - 1)/2) * d
        padding2 = int((k2 - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (k1, k2), stride=stride, padding=(padding1, padding2), bias=False, dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, _, _ = y.size()
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_size, out_size, kernel_size, stride=1):
        super(Block, self).__init__()
        self.stride = stride
        expand_size = in_size // 2
        if isinstance(kernel_size, tuple) and len(kernel_size)==2:
            k1,k2 = kernel_size
        elif isinstance(kernel_size, int):
            k1 = k2 = kernel_size
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(in_size, expand_size, kernel_size=kernel_size, stride=stride, padding=(k1//2, k2//2), groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = AconC(expand_size)
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_size)


    def forward(self, x):
        out = x
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        return out

class AICV2(nn.Module):
    """docstring for AIC.It can be embeded into any conv net"""
    def __init__(self,in_channel,hidden_channel):
        """
        the default setting: in_channel == 2 * hidden_channel
        """
        super(AICV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=3, padding=1)
        # self.x1 = Block(hidden_channel, hidden_channel, (1,3))
        # self.x2 = Block(hidden_channel, hidden_channel, (1,5))
        # self.x3 = Block(hidden_channel, hidden_channel, (1,7))
        # self.y1 = Block(hidden_channel, hidden_channel, (3,1))
        # self.y2 = Block(hidden_channel, hidden_channel, (5,1))
        # self.y3 = Block(hidden_channel, hidden_channel, (7,1))
        
        self.x1 = Dilatedconvolution(hidden_channel, hidden_channel, (1,1), d=1)
        self.x2 = Dilatedconvolution(hidden_channel, hidden_channel, (1,3), d=2)
        self.x3 = Dilatedconvolution(hidden_channel, hidden_channel, (1,5), d=3)
        self.y1 = Dilatedconvolution(hidden_channel, hidden_channel, (1,1), d=1)
        self.y2 = Dilatedconvolution(hidden_channel, hidden_channel, (3,1), d=2)
        self.y3 = Dilatedconvolution(hidden_channel, hidden_channel, (5,1), d=3)

        self.se1= SELayer(hidden_channel, reduction=4)
        self.se2= SELayer(hidden_channel, reduction=4)
        
        self.conv2 = nn.Conv2d(hidden_channel, in_channel, kernel_size=3, padding=1)
        self.weights =  nn.Conv2d(hidden_channel, 6, kernel_size=1)

        self.activate1 = AconC(hidden_channel)
        self.activate2 = AconC(hidden_channel)
        self.activate3 = AconC(in_channel)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = Mish()
        self.bn = nn.InstanceNorm2d(hidden_channel)
        self.softmax = torch.softmax
        self.bn2 = nn.InstanceNorm2d(in_channel)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        tmp = x
        w = self.weights(x)
        wx = self.softmax(w[:,:3],1)
        wy = self.softmax(w[:,3:],1)
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        # print(wx[:,0].shape)
        # print(x1.transpose(0,1).mul(wx[:,0]).shape)
        x = x1.transpose(0,1).mul(wx[:,0]) + x2.transpose(0,1).mul(wx[:,1]) + x3.transpose(0,1).mul(wx[:,2])
        x = x.transpose(0,1)
        x = self.se1(x, tmp)
        x = self.bn(x)
        x = self.activate1(x)
        y1 = self.y1(x)
        y2 = self.y2(x)
        y3 = self.y3(x)
        x = y1.transpose(0,1).mul(wy[:,0])+y2.transpose(0,1).mul(wy[:,1])+y3.transpose(0,1).mul(wy[:,2])
        x = x.transpose(0,1)
        x = self.se2(x, tmp)
        x = self.bn(x)
        x = self.activate2(x)
        x = self.conv2(x)
        x +=residual
        x = self.bn2(x)
        x = self.activate3(x)
        return x


class AIC(nn.Module):
    """docstring for AIC.It can be embeded into any conv net"""
    def __init__(self,in_channel,hidden_channel):
        """
        the default setting: in_channel == 2 * hidden_channel
        """
        super(AIC, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=3, padding=1)
        self.x1 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[1,3], padding=[0,1])
        self.x2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[1,5], padding=[0,2])
        self.x3 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[1,7], padding=[0,3])
        self.y1 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[3,1], padding=[1,0])
        self.y2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[5,1], padding=[2,0])
        self.y3 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=[7,1], padding=[3,0])
        self.conv2 = nn.Conv2d(hidden_channel, in_channel, kernel_size=3, padding=1)
        self.weights =  nn.Conv2d(hidden_channel, 6, kernel_size=1)

        self.activate1 = AconC(hidden_channel)
        self.activate2 = AconC(hidden_channel)
        self.activate3 = AconC(in_channel)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = Mish()
        self.bn = nn.InstanceNorm2d(hidden_channel)
        self.softmax = torch.softmax
        self.bn2 = nn.InstanceNorm2d(in_channel)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        w = self.weights(x)
        wx = self.softmax(w[:,:3],1)
        wy = self.softmax(w[:,3:],1)
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        # print(wx[:,0].shape)
        # print(x1.transpose(0,1).mul(wx[:,0]).shape)
        x = x1.transpose(0,1).mul(wx[:,0]) + x2.transpose(0,1).mul(wx[:,1]) + x3.transpose(0,1).mul(wx[:,2])
        x = x.transpose(0,1)
        x = self.bn(x)
        x = self.activate1(x)
        y1 = self.y1(x)
        y2 = self.y2(x)
        y3 = self.y3(x)
        x = y1.transpose(0,1).mul(wy[:,0])+y2.transpose(0,1).mul(wy[:,1])+y3.transpose(0,1).mul(wy[:,2])
        x = x.transpose(0,1)
        x = self.bn(x)
        x = self.activate2(x)
        x = self.conv2(x)
        x +=residual
        x = self.bn2(x)
        x = self.activate3(x)
        return x



