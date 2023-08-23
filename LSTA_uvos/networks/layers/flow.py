import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x**2) - mu_x**2
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y**2) - mu_y**2
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return SSIM

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

def warp_flow(x, flow, use_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    Inputs:
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow

    Returns:
    ouptut: [B, C, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if grid.shape != flow.shape:
        raise ValueError('the shape of grid {0} is not equal to the shape of flow {1}.'.format(grid.shape, flow.shape))
    if x.is_cuda:
        grid = grid.to(x.get_device())
    vgrid = grid + flow

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    if use_mask:
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.get_device())
        mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask
    else:
        return output

class FlowBranch(nn.Module):
    def __init__(self, md=4):
        super(FlowBranch, self).__init__()
        self.corr = self.corr_naive
        self.leakyRELU = nn.LeakyReLU(0.1)
        nd = (2*md+1)**2
        dd = np.array([128,128,96,64,32])
        feat_dims = [2048,1024,512,256]
        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv6_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = self.predict_flow(dd[3]+dd[4])
        
        od = nd+feat_dims[1]+2
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv5_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = self.predict_flow(dd[3]+dd[4]) 
        
        od = nd+feat_dims[2]+2
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv4_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = self.predict_flow(dd[3]+dd[4]) 
        
        od = nd+feat_dims[3]+2
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(dd[0],   128, kernel_size=3, stride=1)
        self.conv3_2 = conv(dd[0]+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(dd[1]+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(dd[2]+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = self.predict_flow(dd[3]+dd[4])

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_zeros_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        '''
    def predict_flow(self, in_planes):
        return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

    def warp(self, x, flow):
        return warp_flow(x, flow, use_mask=False)

    def corr_naive(self, input1, input2, d=4):
        # naive pytorch implementation of the correlation layer.
        assert (input1.shape == input2.shape)
        batch_size, feature_num, H, W = input1.shape[0:4]
        input2 = F.pad(input2, (d,d,d,d), value=0)
        cv = []
        for i in range(2 * d + 1):
            for j in range(2 * d + 1):
                cv.append((input1 * input2[:, :, i:(i + H), j:(j + W)]).mean(1).unsqueeze(1))
        return torch.cat(cv, 1)
    
    def forward(self, feature_list_1, feature_list_2, img_hw):
        c13, c14, c15, c16 = feature_list_1
        c23, c24, c25, c26 = feature_list_2

        corr6 = self.corr(c16, c26) 
        x0 = self.conv6_0(corr6)
        x1 = self.conv6_1(x0)
        x2 = self.conv6_2(torch.cat((x0,x1),1))
        x3 = self.conv6_3(torch.cat((x1,x2),1))
        x4 = self.conv6_4(torch.cat((x2,x3),1))
        flow6 = self.predict_flow6(torch.cat((x3,x4),1))
        up_flow6 = flow6  # 1/8
        
        warp5 = self.warp(c25, up_flow6)
        corr5 = self.corr(c15, warp5)
        x = torch.cat((corr5, c15, up_flow6), 1)
        x0 = self.conv5_0(x)
        x1 = self.conv5_1(x0)
        x2 = self.conv5_2(torch.cat((x0,x1),1))
        x3 = self.conv5_3(torch.cat((x1,x2),1))
        x4 = self.conv5_4(torch.cat((x2,x3),1))
        flow5 = self.predict_flow5(torch.cat((x3,x4),1))
        flow5 = flow5 + up_flow6
        up_flow5 = F.interpolate(flow5, c24.shape[-2:], mode='bilinear', align_corners=True)*2.0

        warp4 = self.warp(c24, up_flow5)
        corr4 = self.corr(c14, warp4)  
        x = torch.cat((corr4, c14, up_flow5), 1)
        x0 = self.conv4_0(x)
        x1 = self.conv4_1(x0)
        x2 = self.conv4_2(torch.cat((x0,x1),1))
        x3 = self.conv4_3(torch.cat((x1,x2),1))
        x4 = self.conv4_4(torch.cat((x2,x3),1))
        flow4 = self.predict_flow4(torch.cat((x3,x4),1))
        flow4 = flow4 + up_flow5
        up_flow4 = F.interpolate(flow4, c23.shape[-2:], mode='bilinear', align_corners=True)*2.0
        
        warp3 = self.warp(c23, up_flow4)
        corr3 = self.corr(c13, warp3) 
        x = torch.cat((corr3, c13, up_flow4), 1)
        x0 = self.conv3_0(x)
        x1 = self.conv3_1(x0)
        x2 = self.conv3_2(torch.cat((x0,x1),1))
        x3 = self.conv3_3(torch.cat((x1,x2),1))
        x4 = self.conv3_4(torch.cat((x2,x3),1))
        flow3 = self.predict_flow3(torch.cat((x3,x4),1))
        flow3 = flow3 + up_flow4

        img_h, img_w = img_hw[0], img_hw[1]

        flow3 = F.interpolate(flow3 * 4.0, [img_h, img_w ], mode='bilinear', align_corners=True)
        flow4 = F.interpolate(flow4 * 4.0, [img_h // 2, img_w // 2], mode='bilinear', align_corners=True)
        flow5 = F.interpolate(flow5 * 4.0, [img_h // 4, img_w // 4], mode='bilinear', align_corners=True)
        
        return [flow3, flow4, flow5]



class FlowPrediction(nn.Module):
    def __init__(self):
        super(FlowPrediction,self).__init__()
        self.num_scales = 3
        self.flow_pred = FlowBranch()

    def get_flow_normalization(self, flow, p=2):
        '''
        Inputs:
        flow (bs, 2, H, W)
        '''
        flow_norm = torch.norm(flow, p=p, dim=1).unsqueeze(1) + 1e-12
        flow_normalization = flow / flow_norm.repeat(1,2,1,1)
        return flow_normalization

    def generate_img_pyramid(self, img, num_pyramid):
        img_h, img_w = img.shape[2], img.shape[3]
        img_pyramid = []
        for s in range(num_pyramid):
            img_new = F.adaptive_avg_pool2d(img, [int(img_h / (2**s)), int(img_w / (2**s))]).data
            img_pyramid.append(img_new)
        return img_pyramid

    def warp_flow_pyramid(self, img_pyramid, flow_pyramid):
        img_warped_pyramid = []
        for img, flow in zip(img_pyramid, flow_pyramid):
            img_warped_pyramid.append(warp_flow(img, flow, use_mask=True))
        return img_warped_pyramid

    def compute_loss_with_mask(self, diff_list, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            diff, occ_mask = diff_list[scale], occ_mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            img_diff = diff * occ_mask.repeat(1,3,1,1)
            loss_pixel = img_diff.mean((1,2,3)) / (divider + 1e-12) # (B)
            loss_list.append(loss_pixel[:,None])
        loss = torch.cat(loss_list, 1).sum(1) # (B)
        return loss

    def compute_diff_weight(self, img_pyramid_from_l, img_pyramid, img_pyramid_from_r):
        diff_fwd = []
        diff_bwd = []
        weight_fwd = []
        weight_bwd = []
        valid_bwd = []
        valid_fwd = []
        for scale in range(self.num_scales):
            img_from_l, img, img_from_r = img_pyramid_from_l[scale], img_pyramid[scale], img_pyramid_from_r[scale]
            
            valid_pixels_fwd = 1 - (img_from_r == 0).prod(1, keepdim=True).type_as(img_from_r)
            valid_pixels_bwd = 1 - (img_from_l == 0).prod(1, keepdim=True).type_as(img_from_l)

            valid_bwd.append(valid_pixels_bwd)
            valid_fwd.append(valid_pixels_fwd)

            img_diff_l = torch.abs((img-img_from_l)).mean(1, True)
            img_diff_r = torch.abs((img-img_from_r)).mean(1, True)

            diff_cat = torch.cat((img_diff_l, img_diff_r),1)
            weight = 1 - nn.functional.softmax(diff_cat,1)
            weight = Variable(weight.data,requires_grad=False)

            # weight = (weight > 0.48).float()

            weight = 2*torch.exp(-(weight-0.5)**2/0.03)

            weight_bwd.append(torch.unsqueeze(weight[:,0,:,:],1) * valid_pixels_bwd)
            weight_fwd.append(torch.unsqueeze(weight[:,1,:,:],1) * valid_pixels_fwd)

            diff_fwd.append(img_diff_r)
            diff_bwd.append(img_diff_l)
             
        return diff_bwd, diff_fwd, weight_bwd, weight_fwd

    def compute_loss_ssim(self, img_pyramid, img_warped_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            img, img_warped, occ_mask = img_pyramid[scale], img_warped_pyramid[scale], occ_mask_list[scale]
            divider = occ_mask.mean((1,2,3))
            occ_mask_pad = occ_mask.repeat(1,3,1,1)
            ssim = SSIM(img * occ_mask_pad, img_warped * occ_mask_pad)
            loss_ssim = torch.clamp((1.0 - ssim) / 2.0, 0, 1).mean((1,2,3))
            loss_ssim = loss_ssim / (divider + 1e-12)
            loss_list.append(loss_ssim[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

    def gradients(self, img):
        dy = img[:,:,1:,:] - img[:,:,:-1,:]
        dx = img[:,:,:,1:] - img[:,:,:,:-1]
        return dx, dy

    def cal_grad2_error(self, flow, img):
        img_grad_x, img_grad_y = self.gradients(img)
        w_x = torch.exp(-10.0 * torch.abs(img_grad_x).mean(1).unsqueeze(1))
        w_y = torch.exp(-10.0 * torch.abs(img_grad_y).mean(1).unsqueeze(1))

        dx, dy = self.gradients(flow)
        dx2, _ = self.gradients(dx)
        _, dy2 = self.gradients(dy)
        error = (w_x[:,:,:,1:] * torch.abs(dx2)).mean((1,2,3)) + (w_y[:,:,1:,:] * torch.abs(dy2)).mean((1,2,3))
        #error = (w_x * torch.abs(dx)).mean((1,2,3)) + (w_y * torch.abs(dy)).mean((1,2,3))
        return error / 2.0

    def compute_loss_flow_smooth(self, optical_flows, img_pyramid):
        loss_list = []
        for scale in range(self.num_scales):
            flow, img = optical_flows[scale], img_pyramid[scale]
            #error = self.cal_grad2_error(flow, img)
            error = self.cal_grad2_error(flow/20.0, img)
            loss_list.append(error[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

    def compute_loss_flow_consis(self, fwd_flow_pyramid, bwd_flow_pyramid, occ_mask_list):
        loss_list = []
        for scale in range(self.num_scales):
            fwd_flow, bwd_flow, occ_mask = fwd_flow_pyramid[scale], bwd_flow_pyramid[scale], occ_mask_list[scale]
            fwd_flow_norm = self.get_flow_normalization(fwd_flow)
            bwd_flow_norm = self.get_flow_normalization(bwd_flow)
            bwd_flow_norm = Variable(bwd_flow_norm.data,requires_grad=False)
            occ_mask = 1-occ_mask

            divider = occ_mask.mean((1,2,3))
            
            loss_consis = (torch.abs(fwd_flow_norm+bwd_flow_norm) * occ_mask).mean((1,2,3))
            loss_consis = loss_consis / (divider + 1e-12)
            loss_list.append(loss_consis[:,None])
        loss = torch.cat(loss_list, 1).sum(1)
        return loss

    def inference_flow(self, img1, img2):
        img_hw = [img1.shape[2], img1.shape[3]]
        feature_list_1, feature_list_2 = self.fpyramid(img1), self.fpyramid(img2)
        optical_flow = self.pwc_model(feature_list_1, feature_list_2, img_hw)[0]
        return optical_flow
    
    def forward(self, prev_features, current_features, post_features, prev_img, curr_img, post_img):
        imgl,img,imgr = prev_img, curr_img, post_img
        img_h,img_w = curr_img.shape[-2:]
        feature_list_l, feature_list, feature_list_r = prev_features, current_features, post_features
        optical_flows_bwd = self.flow_pred(feature_list, feature_list_l, [img_h, img_w])
        optical_flows_fwd = self.flow_pred(feature_list, feature_list_r, [img_h, img_w])

        loss_pack = {}
        imgl_pyramid = self.generate_img_pyramid(imgl, len(optical_flows_fwd))
        img_pyramid = self.generate_img_pyramid(img, len(optical_flows_fwd))
        imgr_pyramid = self.generate_img_pyramid(imgr, len(optical_flows_fwd))

        img_warped_pyramid_from_l = self.warp_flow_pyramid(imgl_pyramid, optical_flows_bwd)
        img_warped_pyramid_from_r = self.warp_flow_pyramid(imgr_pyramid, optical_flows_fwd)


        diff_bwd, diff_fwd, weight_bwd, weight_fwd = self.compute_diff_weight(img_warped_pyramid_from_l, img_pyramid, img_warped_pyramid_from_r)
        loss_pack['loss_pixel'] = self.compute_loss_with_mask(diff_fwd, weight_fwd) + \
            self.compute_loss_with_mask(diff_bwd, weight_bwd)
        
        loss_pack['loss_ssim'] = self.compute_loss_ssim(img_pyramid, img_warped_pyramid_from_r, weight_fwd) + \
            self.compute_loss_ssim(img_pyramid, img_warped_pyramid_from_l,weight_bwd)

        loss_pack['loss_flow_smooth'] = self.compute_loss_flow_smooth(optical_flows_fwd, img_pyramid)  + \
            self.compute_loss_flow_smooth(optical_flows_bwd, img_pyramid)
        
        loss_pack['loss_flow_consis'] = self.compute_loss_flow_consis(optical_flows_fwd, optical_flows_bwd, weight_fwd)

        return loss_pack


if __name__ == "__main__":
    H,W = 480,854
    img = torch.rand(1,3,H,W)
    feat1 = torch.rand(1,256,H//2,W//2)
    feat2 = torch.rand(1,512,H//4,W//4)
    feat3 = torch.rand(1,1024,H//8,W//8)
    feat4 = torch.rand(1,2048,H//8,W//8)
    feat_list = [feat1,feat2,feat3,feat4]
    model = FlowPrediction()
    loss_pack = model(feat_list, feat_list, feat_list, img, img, img)
    # loss_pixel', 'loss_ssim', 'loss_flow_smooth', 'loss_flow_consis
# w_ssim: 0.85 # w_pixel = 1 - w_ssim
# w_flow_smooth: 10.0
# w_flow_consis: 0.01
# w_geo: 1.0
# w_pt_depth: 1.0
# w_pj_depth: 0.1
# w_flow_error: 0.0
# w_depth_smooth: 0.001


# def generate_loss_weights_dict(cfg):
#     weight_dict = {}
#     weight_dict['loss_pixel'] = 1 - cfg.w_ssim
#     weight_dict['loss_ssim'] = cfg.w_ssim
#     weight_dict['loss_flow_smooth'] = cfg.w_flow_smooth
#     weight_dict['loss_flow_consis'] = cfg.w_flow_consis
#     return weight_dict
    loss_list = []
    loss_weights_dict ={
        'loss_pixel': 1- 0.85,
        'loss_ssim':0.85,
        'loss_flow_smooth':10.0,
        'loss_flow_consis':0.01
    }
    for key in list(loss_pack.keys()):
        loss_list.append((loss_weights_dict[key] * loss_pack[key].mean()).unsqueeze(0))
    loss = torch.cat(loss_list, 0).sum()