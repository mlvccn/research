import torch
from torch import nn
import torch.nn.functional as F

# import model.resnet as models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import vgg16_bn, vgg19_bn
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResUnet(nn.Module):
    def __init__(self, 
                 layers=18, 
                 out_dims=[48, 64, 128, 256],
                 classes=2, 
                 BatchNorm=nn.BatchNorm2d, 
                 pretrained=True):
        super(ResUnet, self).__init__()
        assert classes > 1
        if layers == 18:
            resnet = resnet18(pretrained=pretrained)
            # out_dims = [96, 128, 256, 512]
            # out_dims = [48, 64, 128, 256]
        elif layers == 34:
            resnet = resnet34(pretrained=pretrained)
            # in_dims = [64, 128, 256, 512]
            # out_dims = [96, 128, 256, 512]
            # out_dims = [48, 64, 128, 256]
        elif layers == 50:
            resnet = resnet50(pretrained=pretrained)
            # in_dims = [256, 512, 1024, 2048]
            # out_dims = [96, 128, 256, 512]
            # out_dims = [96, 256, 256, 256]
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained)
            # in_dims = [256, 512, 1024, 2048]
        elif layers == 152:
            resnet = resnet152(pretrained=pretrained)
            # in_dims = [256, 512, 1024, 2048]
        elif layers == 16:
            encoder = vgg16_bn(pretrained=pretrained)
            # in_dims = [128, 256, 512, 512]
        elif layers == 19:
            encoder = vgg19_bn(pretrained=pretrained)
            # in_dims = [128, 256, 512, 512]
        
        if layers == 16:
            i = 0
            module = []
            while(i < 7):
                module.append(encoder.features[i])
                i+=1
            self.layer0 = nn.Sequential(*module)

            module = []
            while(i < 14):
                module.append(encoder.features[i])
                i+=1
            self.layer1 = nn.Sequential(*module)

            module = []
            while(i < 24):
                module.append(encoder.features[i])
                i+=1
            self.layer2 = nn.Sequential(*module)

            module = []
            while(i < 34):
                module.append(encoder.features[i])
                i+=1
            self.layer3 = nn.Sequential(*module)

            module = []
            while(i < 44):
                module.append(encoder.features[i])
                i+=1
            self.layer4 = nn.Sequential(*module)
      
        elif layers == 19:
            i = 0
            module = []
            while(i < 7):
                module.append(encoder.features[i])
                i+=1
            self.layer0 = nn.Sequential(*module)

            module = []
            while(i < 14):
                module.append(encoder.features[i])
                i+=1
            self.layer1 = nn.Sequential(*module)

            module = []
            while(i < 27):
                module.append(encoder.features[i])
                i+=1
            self.layer2 = nn.Sequential(*module)

            module = []
            while(i < 40):
                module.append(encoder.features[i])
                i+=1
            self.layer3 = nn.Sequential(*module)

            module = []
            while(i < 53):
                module.append(encoder.features[i])
                i+=1
            self.layer4 = nn.Sequential(*module)
        else:
            block = BasicBlock
            layers = [1, 1, 1, 1]
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            # Decoder
            self.up4 = nn.Sequential(
                nn.Conv2d(out_dims[-1], out_dims[-2], kernel_size=3, stride=1, padding=1), 
                BatchNorm(out_dims[-2]),
                nn.ReLU(inplace=True)
            )
            resnet.inplanes = out_dims[-2] + out_dims[-2]
            self.delayer4 = resnet._make_layer(block, out_dims[-2], layers[-1])

            self.up3 = nn.Sequential(
                nn.Conv2d(out_dims[-2], out_dims[-3], kernel_size=3, stride=1, padding=1), 
                BatchNorm(out_dims[-3]), 
                nn.ReLU(inplace=True)
            )
            resnet.inplanes = out_dims[-3] + out_dims[-3]
            self.delayer3 = resnet._make_layer(block, out_dims[-3], layers[-2])

            self.up2 = nn.Sequential(
                nn.Conv2d(out_dims[-3], out_dims[-4], kernel_size=3, stride=1, padding=1), 
                BatchNorm(out_dims[-4]), 
                nn.ReLU(inplace=True)
            )
            resnet.inplanes = out_dims[-4] + out_dims[-4]
            self.delayer2 = resnet._make_layer(block, out_dims[-4], layers[-3])

            self.cls = nn.Sequential(
                nn.Conv2d(out_dims[-4], out_dims[-4], kernel_size=3, padding=1, bias=False),
                BatchNorm(out_dims[-4]),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(out_dims[-4], classes, kernel_size=1)
            )

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.layer0(x)  # 1/4
        x2 = self.layer1(x)  # 1/4
        x3 = self.layer2(x2)  # 1/8
        x4 = self.layer3(x3)  # 1/16
        x5 = self.layer4(x4)  # 1/32
        p4 = self.up4(F.interpolate(x5, x4.shape[-2:], mode='bilinear', align_corners=True))
        p4 = torch.cat([p4, x4], dim=1)
        p4 = self.delayer4(p4)
        p3 = self.up3(F.interpolate(p4, x3.shape[-2:], mode='bilinear', align_corners=True))
        p3 = torch.cat([p3, x3], dim=1)
        p3 = self.delayer3(p3)
        p2 = self.up2(F.interpolate(p3, x2.shape[-2:], mode='bilinear', align_corners=True))
        p2 = torch.cat([p2, x2], dim=1)
        p2 = self.delayer2(p2)
        x = self.cls(p2)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x

if __name__ == '__main__':
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(1)
    unet = ResUnet(layers=19, out_dims=[64, 128, 256, 512], classes=9).cuda(1)
    unet.eval()
    out = unet(rgb)
    print(unet)
    print("The output size: {}".format(out.size()))
