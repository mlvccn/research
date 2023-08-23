import torch.nn as nn
from torch.nn import functional as F
import torch
affine_par = True

from torch.nn import SyncBatchNorm as norm

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    # output stride 16: (6, 12, 18)
    # output stride 8: (12, 24, 36)
    def __init__(self, features, hidden_features=512, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=1, bias=False),
                                   norm(hidden_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   norm(hidden_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   norm(hidden_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, hidden_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   norm(hidden_features))
        self.image_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, hidden_features, kernel_size=1, bias=False),
                                   norm(hidden_features))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(hidden_features * 5, out_features, kernel_size=1, bias=False),
            norm(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):
        _, _, h, w = x.size()
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        img_feat = F.interpolate(self.image_pooling(x), size=(h, w), mode='bilinear', align_corners=True)
        concat_feat = torch.cat((feat1, feat2, feat3, feat4, img_feat), 1)

        out = self.conv_bn_dropout(concat_feat)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1,2,4))
        
        self.head = nn.Sequential(ASPPModule(2048),
            nn.Conv2d(512, num_classes, kernel_size=1)
            )
        
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            norm(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
            )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)
        return [x, x_dsn]


def ResNetDeepLabv3(backbone='ResNet50', num_classes=1, batch_mode='sync'):
    global BatchNorm2d
    if batch_mode=='sync':
        BatchNorm2d = norm
    elif batch_mode=='old':
        BatchNorm2d = torch.nn.BatchNorm2d

    if backbone=='ResNet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    elif backbone=='ResNet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    else:
        raise RuntimeError('unknown backbone type')
    return model

