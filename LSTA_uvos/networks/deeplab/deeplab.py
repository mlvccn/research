import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.deeplab.aspp import build_aspp
from networks.deeplab.decoder import build_decoder
from networks.deeplab.backbone import build_backbone
from networks.layers.normalization import FrozenBatchNorm2d

class DeepLab(nn.Module):
    def __init__(self, 
            backbone='resnet', 
            num_classes=128,
            output_stride=16,
            batch_mode=None,
            freeze_bn=True):
        super(DeepLab, self).__init__()
        # DeepLab(backbone, num_classes=embedding, batch_mode=batch_mode)
        self.emb_dim = num_classes
        if freeze_bn == True:
            print("Use frozen BN in DeepLab!")
            BatchNorm = FrozenBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(backbone, BatchNorm)
        # self.projection_k = nn.Conv2d(256,num_classes,1)
        # self.projection_v = nn.Conv2d(256,num_classes*2,1)
        

    def forward(self, input):
        feaure_layer1,feaure_layer2,feaure_layer3,feaure_layer4 = self.backbone(input)
        x = feaure_layer4
        low_level_feat = feaure_layer1
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        # x = self.projection(x)

        # return (x_k,x_v),None
        return x, (feaure_layer1,feaure_layer2,feaure_layer3,feaure_layer4)


    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    input = torch.rand(2, 3, 513, 513)
    output = model(input)
    print(output.size())


