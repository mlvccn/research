from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import random

def l2_norm(v, dim=1):
    fnorm = torch.norm(v, p=2, dim=dim, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v


def object_kernel_loss(hs, alpha=0.005):
    """[summary]

    Args:
        hs ([type]): [description]
        alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

    Returns:
        [type]: [description]
    """
    t,b,n,c = hs.shape
    hs = rearrange(hs, 't b n c-> (t b ) n c')
    hs = l2_norm(hs, dim=-1)
    score_map = torch.bmm(hs, hs.transpose(2, 1))
    ltb = t * b
    identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
    loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
    loss = loss.view(t, b, 1, 1)
    # loss = torch.mean(loss)
    return loss * alpha


# def object_kernel_loss(hs, alpha=0.005):
#     """[summary]

#     Args:
#         hs ([type]): [description]
#         alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

#     Returns:
#         [type]: [description]
#     """
#     t,b,n,c = hs.shape
#     hs = rearrange(hs, 't b n c-> (t b ) n c')
#     score_map = torch.bmm(hs, hs.transpose(2, 1)) / math.sqrt(c)
#     ltb = t * b
#     identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
#     loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
#     loss = loss.view(t, b, 1, 1)
#     # loss = torch.mean(loss)
#     return loss * alpha

def object_kernel_loss_w_norm(hs):
    """[summary]

    Args:
        hs ([type]): [description]
        alpha (float, optional): [description]. Defaults to 0.001, value of loss is about 0.5

    Returns:
        [type]: [description]
    """
    t,b,n,c = hs.shape
    hs = rearrange(hs, 't b n c-> (t b ) n c')
    hsnorm = torch.norm(hs, p=2, dim=-1, keepdim=True) + 1e-6
    hs = hs.div(hsnorm.expand_as(hs))
    score_map = torch.bmm(hs, hs.transpose(2, 1)) # tb * n * n
    # print(score_map.shape)
    # print(torch.max(score_map), torch.min(score_map))
    ltb = t * b
    identiy_matrix = torch.eye(n).unsqueeze(0).repeat(ltb, 1, 1).to(hs.device)
    loss = torch.norm(score_map-identiy_matrix, p='fro', dim=(1,2), keepdim=True)
    loss = loss.view(t, b, 1, 1)
    # loss = torch.mean(loss)
    return loss


def grid_sampling(feature_map, sample_numbers):
    """[divide feature map into grids, sample feature vectors from each grid]

    Args:
        feature_map ([tensor]): [shape is (b, c, h, w)]
        sample_numbers ([list]): [length is 2. First is x elements from w; second is y elements from h]

    Returns:
        sampled_feats [tensor]: [shape is (b, y*x, c)]
    """
    b, c, h, w = feature_map.shape
    x, y = sample_numbers
    xs, ys = w//x, h//y
    x_sample = torch.arange(0, x * xs, xs).view(1, 1, x)
    x_sample = x_sample + torch.randint(0, xs, (b, 1, 1))

    y_sample = torch.arange(0, y * ys, ys).view(1, y, 1)
    y_sample = y_sample + torch.randint(0, ys, (b, 1, 1))
    hw_index = torch.LongTensor(x_sample + y_sample * w).to(feature_map.device) # [b, y, x]
    raw_index = hw_index.view(b,-1,1)
    hw_index = hw_index.view(b,-1,1).expand(-1,-1,c) # [b, y*x, c]
    feature_map = rearrange(feature_map, 'b c h w-> b (h w ) c')
    sampled_feats = torch.gather(feature_map, 1,  hw_index) # [b, y*x, c]
    return sampled_feats, raw_index

def affine_forward(a, b):
    """[for each element in a, find the most similar element in b]

    Args:
        a ([tensor]): [shape is (b, h*w, c)]
        b ([tensor]): [shape is (b, h*w, c)]
    """
    c = a.shape[-1]
    # 1. calculate affinity map
    anorm = torch.norm(a, p=2, dim=-1, keepdim=True) + 1e-6
    a = a.div(anorm.expand_as(a))
    bnorm = torch.norm(b, p=2, dim=-1, keepdim=True) + 1e-6
    b = b.div(bnorm.expand_as(b))
    
    affinity_map_a2b = torch.bmm(a, b.transpose(2, 1))
    affinity_map_a2b = torch.softmax(affinity_map_a2b, dim=-1)
    # 2. find most similar index
    forward_index = affinity_map_a2b.argmax(dim=-1) # current frame feature
    raw_index = forward_index.unsqueeze(-1)
    forward_index = forward_index.unsqueeze(-1).expand(-1, -1, c)
    sampled_b = torch.gather(b, 1,forward_index)

    return sampled_b, raw_index

def cycle_loss(frame_features):
    # [L, T, B, N, H_mask, W_mask]
    h_n, w_n=3, 4
    T = frame_features[0]
    cur_feat = frame_features[0]
    cur_feat_sampled = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
    start_feat_sampled = cur_feat_sampled
    # foward pass
    for i in range(1, T):
        nxt_feat = rearrange(frame_features[i], 'b c h w-> b (h w) c') 
        nxt_feat_sampled = affine_forward(cur_feat_sampled, nxt_feat)
        cur_feat_sampled = nxt_feat_sampled
    
    # backward pass
    for i in range(T-1, 0, -1):
        prev_feat = rearrange(frame_features[i-1], 'b c h w-> b (h w) c') 
        prev_feat_sampled = affine_forward(cur_feat_sampled, prev_feat)
        cur_feat_sampled = prev_feat_sampled
    cycle_back_feat_sample = cur_feat_sampled
    loss = F.mse_loss(start_feat_sampled, cycle_back_feat_sample)
    return loss

def triplet_cycle_loss(frame_features, gt, margin=0.4):
    """[summary]

    Args:
        frame_features ([type]): [shape is (t b c h w)]
        gt ([type]): [shape is (t * b * o h w)]
        margin (float, optional): [description]. Defaults to 0.4.

    Returns:
        [type]: [description]
    """
    # [T, B, N, H_mask, W_mask]
    # t b d h w
    # print(frame_features.shape, gt.shape)
    t, b, _, h, w = frame_features.shape
    gt = F.interpolate(gt.unsqueeze(1).float(), size=(h,w), mode='nearest').squeeze(1).view(t,b,-1, h,w)
    # print(gt.shape)
    h_n, w_n=4, 4
    T = frame_features.shape[0]
    cur_feat = frame_features[0]
    cur_feat_sampled, start_index = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
    anchor_feat_sampled, anchor_index = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
    start_feat_sampled = cur_feat_sampled
    # foward pass
    for i in range(1, T):
        nxt_feat = rearrange(frame_features[i], 'b c h w-> b (h w) c') 
        # print(cur_feat_sampled.shape, nxt_feat.shape)
        nxt_feat_sampled,_ = affine_forward(cur_feat_sampled, nxt_feat)
        cur_feat_sampled = nxt_feat_sampled
    
    # backward pass
    for i in range(T-1, 0, -1):
        prev_feat = rearrange(frame_features[i-1], 'b c h w-> b (h w) c')
        prev_feat_sampled,back_index = affine_forward(cur_feat_sampled, prev_feat)
        cur_feat_sampled = prev_feat_sampled
    cycle_back_feat_sample = cur_feat_sampled
    # cycle_index = back_index
    # print(start_feat_sampled.shape)
    # loss = F.mse_loss(start_feat_sampled, cycle_back_feat_sample)
    B, N = start_feat_sampled.shape[:2]
    loss = 0
    gt = rearrange(gt[0], 'b c h w-> b (h w) c')
    # print(gt.shape,start_index.shape)
    # print(start_index.shape)
    # print(start_index.repeat(1, 1, 3).shape)
    # print(gt.shape[-1])
    # print(start_index.repeat(1, 1, gt.shape[-1]).shape)
    start_sampled_gt = torch.gather(gt, 1,  start_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
    anchor_sampled_gt = torch.gather(gt, 1,  anchor_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
    # cycled_gt = torch.gather(gt, 1,  cycle_index[0:,:,0:1]) # [b, y*x, c]
    for b in range(B):
        for n in range(N):
            start_vector = start_feat_sampled[b,n].unsqueeze(0)
            cycle_vector = cycle_back_feat_sample[b,n].unsqueeze(0)
            anchor_vector = anchor_feat_sampled[b,n].unsqueeze(0)
            if all(anchor_sampled_gt[b,n] == start_sampled_gt[b,n] ):
                # print(anchor_sampled_gt[b,n])
                # print(anchor_sampled_gt[b,n], start_sampled_gt[b,n])
                postive = start_vector
                negtive = cycle_vector
            else:
                postive = cycle_vector
                negtive = start_vector
            loss = loss + F.triplet_margin_loss(anchor_vector, postive, negtive, margin=margin)  / (B*N)
    return loss

def triplet_cycyle_loss_v2(frame_features, gt,margin=0.4):
 # [T, B, N, H_mask, W_mask]
    # t b d h w
    # print(frame_features.shape, gt.shape)
    t, b, _, h, w = frame_features.shape
    gt = F.interpolate(gt.unsqueeze(1).float(), size=(h,w), mode='nearest').squeeze(1).view(t,b,-1, h,w)
    # print(gt.shape)
    h_n, w_n=4, 4
    T = frame_features.shape[0]
    cur_feat = frame_features[0]
    cur_feat_sampled, start_index = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
    anchor_feat_sampled, anchor_index = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
    start_feat_sampled = cur_feat_sampled
    # foward pass
    for i in range(1, T):
        nxt_feat = rearrange(frame_features[i], 'b c h w-> b (h w) c') 
        # print(cur_feat_sampled.shape, nxt_feat.shape)
        nxt_feat_sampled,_ = affine_forward(cur_feat_sampled, nxt_feat)
        cur_feat_sampled = nxt_feat_sampled
    
    # backward pass
    for i in range(T-1, 0, -1):
        prev_feat = rearrange(frame_features[i-1], 'b c h w-> b (h w) c')
        prev_feat_sampled,back_index = affine_forward(cur_feat_sampled, prev_feat)
        cur_feat_sampled = prev_feat_sampled
    cycle_back_feat_sample = cur_feat_sampled
    cycle_index = back_index
    # print(start_feat_sampled.shape)
    # loss = F.mse_loss(start_feat_sampled, cycle_back_feat_sample)
    B, N = start_feat_sampled.shape[:2]
    loss = 0
    gt = rearrange(gt[0], 'b c h w-> b (h w) c')
    # print(gt.shape,start_index.shape)
    # print(start_index.shape)
    # print(start_index.repeat(1, 1, 3).shape)
    # print(gt.shape[-1])
    # print(start_index.repeat(1, 1, gt.shape[-1]).shape)
    start_sampled_gt = torch.gather(gt, 1,  start_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
    anchor_sampled_gt = torch.gather(gt, 1,  anchor_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
    # cycled_gt = torch.gather(gt, 1,  cycle_index[0:,:,0:1]) # [b, y*x, c]
    for b in range(B):
        for n in range(N):
            start_vector = start_feat_sampled[b,n].unsqueeze(0)
            cycle_vector = cycle_back_feat_sample[b,n].unsqueeze(0)
            anchor_vector = anchor_feat_sampled[b,n].unsqueeze(0)
            if any(anchor_sampled_gt[b,n] != start_sampled_gt[b,n] ):
                # print(anchor_sampled_gt[b,n])
                # print(anchor_sampled_gt[b,n], start_sampled_gt[b,n])
                anchor = start_vector
                postive = anchor_vector
                negtive = cycle_vector
            else:
                anchor = start_vector
                postive = cycle_vector
                negtive = anchor_vector
            loss = loss + F.triplet_margin_loss(anchor, postive, negtive, margin=margin)  / (B*N)
            
    return loss



class InstanceLoss(nn.Module):
    def __init__(self, gamma = 1) -> None:
        super(InstanceLoss, self).__init__()
        self.gamma = gamma

    def forward(self, feature, label = None):
        # Dual-Path Convolutional Image-Text Embeddings with Instance Loss, ACM TOMM 2020 
        # https://zdzheng.xyz/files/TOMM20.pdf 
        # using cross-entropy loss for every sample if label is not available. else use given label.
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature*self.gamma, torch.t(normed_feature)) 
        # print(sim1.shape)
        #sim2 = sim1.t()
        if label is None:
            sim_label = torch.arange(sim1.size(0)).cuda().detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
            # print(sim_label)
            # print(label)
        loss = F.cross_entropy(sim1, sim_label) #+ F.cross_entropy(sim2, sim_label)

        return loss


def triplet_cycyle_loss_v3(feats, gt, margin=0.3):
    """[summary]

    Args:
        feats ([type]): [description]
        gt ([type]): [description]
        margin (float, optional): [description]. Defaults to 0.3.
    """
    # print()
    # print(feats.shape, gt.shape) # torch.Size([2, 1, 8, 80, 142]) torch.Size([14, 320, 568])
# torch.Size([16, 360, 640])
    # gt = gt.unsqueeze(1).unsqueeze(2) # t,h,w->t,b,1,h,w
    T,B,_,H,W = feats.shape
    # gt = gt.view(T*B,1, *gt.shape[-2:])
    # sample numbers
    h_n, w_n=H//8, W//8
    gt = F.interpolate(gt.unsqueeze(1).float(), size=(H,W), mode='nearest').squeeze(1).view(T,B,-1, H,W).int()

    # gt = F.interpolate(gt.float(), size=(H,W), mode='nearest').squeeze(1).view(T,B,-1, H,W).int()
    # print(gt.shape)
    loss = 0
    gt = rearrange(gt, 't b c h w-> t b (h w) c')

    t = random.randint(1, T-1)
    # for t in range(1, T):
    cur_feat = feats[t-1]
    cur_feat_sampled, start_index = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
    nxt_feat = rearrange(feats[t], 'b c h w-> b (h w) c') 
    # frame t-1  -->   frame t
    nxt_feat_sampled, nxt_index = affine_forward(cur_feat_sampled, nxt_feat)
    # frame t  -->   frame t-1
    cur_feat = rearrange(feats[t-1], 'b c h w-> b (h w) c')
    cyc_feat_sampled, cyc_index = affine_forward(nxt_feat_sampled, cur_feat)
    cur_gt = gt[t-1]
    nxt_gt = gt[t]
    # print(cur_gt.shape,start_index.shape)
    cur_sampled_gt = torch.gather(cur_gt, 1,  start_index.expand(-1, -1, cur_gt.shape[-1])) # [b, y*x, c]
    nxt_sampled_gt = torch.gather(nxt_gt, 1,  nxt_index.expand(-1, -1, nxt_gt.shape[-1])) # [b, y*x, c]
    cyc_sampled_gt = torch.gather(cur_gt, 1,  cyc_index.expand(-1, -1, cur_gt.shape[-1])) # [b, y*x, c]
    
    B, N = cur_feat_sampled.shape[:2]
    
    for b in range(B):
        for n in range(N):
            cur_vector = cur_feat_sampled[b,n].unsqueeze(0)
            nxt_vector = nxt_feat_sampled[b,n].unsqueeze(0)
            cyc_vector = cyc_feat_sampled[b,n].unsqueeze(0)
            gt1 = cur_sampled_gt[b,n]
            gt2 = nxt_sampled_gt[b,n]
            gt3 = cyc_sampled_gt[b,n]
            d1 = 1.0 - F.cosine_similarity(cur_vector, nxt_vector) 
            d2 = 1.0 - F.cosine_similarity(cyc_vector, nxt_vector) 
            d3 = 1.0 - F.cosine_similarity(cur_vector, cyc_vector) 
            if all(gt1 == gt2) and all(gt2 == gt3 ):
                # minimize distance 
                loss = loss + (d1 + d2 + d3) / 3.0
            elif all( gt1 ^ gt2 ^ gt3  == gt1):
                triplet_loss1 = torch.clamp(d2 - d3 + margin, min=0.0).mean()
                triplet_loss2 = torch.clamp(d2 - d1 + margin, min=0.0).mean()
                loss = loss + 0.5 * ( triplet_loss1 + triplet_loss2 )
            elif all( gt1 ^ gt2 ^ gt3  == gt2):
                # negative = nxt_vector
                triplet_loss1 = torch.clamp(d3 - d1 + margin, min=0.0).mean()
                triplet_loss2 = torch.clamp(d3 - d2 + margin, min=0.0).mean()
                loss = loss + 0.5 * ( triplet_loss1 + triplet_loss2 )
            else:
                    # negative = cyc_vector
                triplet_loss1 = torch.clamp(d1 - d2 + margin, min=0.0).mean()
                triplet_loss2 = torch.clamp(d1 - d3 + margin, min=0.0).mean()
                loss = loss + 0.5 * ( triplet_loss1 + triplet_loss2 ) 
    loss = loss / (B*N)

    return loss


def triplet_cycyle_loss_v4(feats, gt, margin=0.3):
    """[summary]

    Args:
        feats ([type]): [description]
        gt ([type]): [description]
        margin (float, optional): [description]. Defaults to 0.3.
    """
    # print(feats.shape, gt.shape)
    # gt = gt.unsqueeze(1).unsqueeze(2) # t,h,w->t,b,1,h,w
    T,B,_,H,W = feats.shape
    gt = gt.view(T*B,-1, *gt.shape[-2:])
    # sample numbers
    h_n, w_n=H//8, W//8
    gt = F.interpolate(gt.float(), size=(H,W), mode='nearest').squeeze(1).view(T,B,-1, H,W).int()

    loss = 0
    gt = rearrange(gt, 't b c h w-> t b (h w) c')

    # gt = rearrange(gt, 't b c h w-> t b (h w) c')
    loss = torch.zeros(T, B).to(gt.device)
    for t in range(1, T):
        cur_feat = feats[t-1]
        cur_feat_sampled, start_index = grid_sampling(cur_feat, sample_numbers=(h_n, w_n))
        nxt_feat = rearrange(feats[t], 'b c h w-> b (h w) c') 
        # frame t-1  -->   frame t
        nxt_feat_sampled, nxt_index = affine_forward(cur_feat_sampled, nxt_feat)
        # frame t  -->   frame t-1
        cur_feat = rearrange(feats[t-1], 'b c h w-> b (h w) c')
        cyc_feat_sampled, cyc_index = affine_forward(nxt_feat_sampled, cur_feat)
        # print(t, gt.shape)
        cur_gt = gt[t-1]
        # cur_gt = rearrange(cur_gt, 'b c h w-> b (h w) c')
        nxt_gt = gt[t]
        # nxt_gt = rearrange(nxt_gt, 'b c h w-> b (h w) c')
        # print(cur_gt.shape, start_index.shape,gt.shape[1])
        cur_sampled_gt = torch.gather(cur_gt, 1,  start_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
        nxt_sampled_gt = torch.gather(nxt_gt, 1,  nxt_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
        cyc_sampled_gt = torch.gather(cur_gt, 1,  cyc_index.expand(-1, -1, gt.shape[-1])) # [b, y*x, c]
        
        B, N = cur_feat_sampled.shape[:2]
        cur_feat_sampled = cur_feat_sampled.view(B*N, -1)
        nxt_feat_sampled = nxt_feat_sampled.view(B*N, -1)
        cyc_feat_sampled = cyc_feat_sampled.view(B*N, -1)
        d1 = 1.0 - F.cosine_similarity(cur_feat_sampled, nxt_feat_sampled).view(B, N)
        d2 = 1.0 - F.cosine_similarity(cyc_feat_sampled, nxt_feat_sampled).view(B, N) 
        d3 = 1.0 - F.cosine_similarity(cur_feat_sampled, cyc_feat_sampled).view(B, N)
        condition1 = (0 == ((cur_sampled_gt != nxt_sampled_gt).sum(dim=-1))).float().unsqueeze(-1)
        condition2 = (0 == ((nxt_sampled_gt != cyc_sampled_gt).sum(dim=-1))).float().unsqueeze(-1)
        condition3 = (0 == ((cur_sampled_gt != cyc_sampled_gt).sum(dim=-1))).float().unsqueeze(-1)

        # loss = loss + (d1 + d2 + d3) / 3.0 * condition1 * condition2 + \
        #     0.5 * condition2 * ( torch.clamp(d2 - d3 + margin, min=0.0).mean() + torch.clamp(d2 - d1 + margin, min=0.0).mean() ) + \
        #     0.5 * condition3 * ( torch.clamp(d3 - d1 + margin, min=0.0).mean() + torch.clamp(d3 - d2 + margin, min=0.0).mean() ) + \
        #     0.5 * condition1 * ( torch.clamp(d1 - d2 + margin, min=0.0).mean() + torch.clamp(d1 - d3 + margin, min=0.0).mean() ) 
        loss1 = ((d1 + d2 + d3) / 3.0 * condition1 * condition2).mean(dim=(-2,-1))
        # print(loss1.shape)
        loss2 = (condition2 * torch.clamp(d2 - d3 + margin, min=0.0)).mean(dim=(-2,-1))
        # print(loss2.shape)
        loss3 = (condition3 * torch.clamp(d3 - d1 + margin, min=0.0)).mean(dim=(-2,-1))
        loss4 = (condition1 * torch.clamp(d1 - d2 + margin, min=0.0)).mean(dim=(-2,-1))
        loss[t] = loss[t] + (loss1 + loss2 + loss3 + loss4)
        # loss[t] = loss[t] + loss4
        # print(loss[t])
        # loss = loss + 
    loss = loss / (T-1)
    return loss.sum()


if __name__ =="__main__":
    hs = torch.rand( 8, 1, 50, 256)
    loss = object_kernel_loss_w_norm(hs)
    loss1 = object_kernel_loss(hs, alpha=0.001)
    loss2 = object_kernel_loss_w_norm(hs)
    print(torch.mean(loss1), torch.mean(loss2) * 0.01)
    
    # feats = torch.rand(8, 1, 256, 32, 48)
    # loss = triplet_cycle_loss(feats)
    # print(loss)

    # x = torch.rand(1, 256)
    # y = torch.rand(1, 256)
    # z = F.cosine_similarity(x, y, dim=1)
    # print(z)

    # feat = F.normalize(torch.rand(20, 64, requires_grad=True))
    # lbl = torch.randint(high=10, size=(20,))
    # criterion = InstanceLoss()
    # instance_loss = criterion(feat, lbl)
    # print(instance_loss/256)

    # feats = torch.rand(8, 1, 256, 32, 48)
    # gt = torch.randint(0, 2, (24, 32, 48)).int()
    # loss = triplet_cycle_loss(feats, gt)
    # print(loss)