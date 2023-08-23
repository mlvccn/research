from networks.deeplab.deeplab import DeepLab
from networks.layers.attention import Performer, TemporalPatchAttentionV9
from networks.layers.esp import AIC
from networks.layers.activation import AconC

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import SyncBatchNorm as norm


class LSTA(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        batch_mode='sync'
        embedding = cfg.MODEL_SEMANTIC_EMBEDDING_DIM
        attn_dim = 16
        if cfg.MODEL_BACKBONE =='resnet':
            self.backbone_dims = {
                'layer1':256,
                'layer2':512,
                'layer3':1024,
                'layer4':2048
            }
        elif cfg.MODEL_BACKBONE=='mobilenet':
            self.backbone_dims ={
            }
        else:
            raise NotImplementedError

        self.feature_extractor = DeepLab(
            cfg.MODEL_BACKBONE, 
            num_classes=embedding,
            batch_mode=batch_mode
        )
        
        self.projection_m = nn.Conv2d(256,attn_dim,1)
        self.projection_curr = nn.Conv2d(256,attn_dim,1)
        self.projection_curr_patch = nn.Conv2d(256,attn_dim,1)

        self.STA = TemporalPatchAttentionV9(attn_dim,attn_dim)
        self.LTM = Performer(
            dim=attn_dim,
            in_dim=attn_dim
        )

        self.refine = nn.Sequential(
            nn.Conv2d(attn_dim*3, embedding, kernel_size=3, stride=1, padding=1),
            norm(embedding),
            AconC(embedding),
            AIC(embedding, embedding)
        )
        self.pred = nn.Conv2d(embedding, 2, kernel_size=3, stride=1, padding=1)
        

    def forward(self, sample):
        img1 = sample['ref_img']
        img2 = sample['prev_img']
        feat1 = self.feature_extractor(img1)[0]
        feat2 = self.feature_extractor(img2)[0]
        memory_features_list = [feat1, feat2]
        pred_list = []
        for i in range(len(sample['curr_img'])):
            img = sample['curr_img'][i]
            feat = self.feature_extractor(img)[0]

            q = self.projection_curr(feat)
            q_patch = self.projection_curr_patch(feat)
            # q_patch = q

            k1 = self.projection_m(memory_features_list[0])
            k2 = self.projection_m(memory_features_list[1])

            k = torch.cat([k1,k2], dim=2)
            x = self.LTM(q,k,k)
            z = self.STA(q_patch, k2,k2)
            x = torch.cat([x,z], dim=1)

            x = self.refine(x)

            pred = self.pred(x)
            
            h,w = img.shape[-2:]
            pred = F.interpolate(pred, (h,w), mode='bilinear', align_corners=True)
            pred_list.append(pred)

            del memory_features_list[0]
            memory_features_list.append(feat)
        
        return pred_list

    def forward_mem(self,reference):
        mem_features = self.feature_extractor(reference)[0]
        return mem_features

    def forward_curr(self, memory_features_list,idx=-1):
        memory_features_list_ = memory_features_list.copy()
        curr_features_ = memory_features_list_.pop(idx)
        features_proj_list = []
        for i in range(len(memory_features_list_)):
            tmp = self.projection_m(memory_features_list_[i])
            features_proj_list.append(tmp)
        curr_features = self.projection_curr(curr_features_)
        curr_features_patch = self.projection_curr_patch(curr_features_)
        # curr_features_patch = curr_features
        
        feat_mems = torch.cat(features_proj_list[:-1], dim=2)
        x = self.LTM(curr_features,feat_mems,feat_mems)
        z = self.STA(curr_features_patch, features_proj_list[-1],features_proj_list[-1])
        x = torch.cat([x,z], dim=1)
        
        x = self.refine(x)

        x = self.pred(x)

        return x

