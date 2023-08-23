import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networks.layers.lovasz_losses as L

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        probability = torch.sigmoid(input)
        batch_size = target.shape[0]
        channel_num = target.shape[1]
        dice = 0.0
        for i in range(batch_size):
            for j in range(channel_num):
                channel_loss = self.dice_loss_single_channel(probability[i, j, :, :], target[i, j, :, :])
                dice += channel_loss / (batch_size * channel_num)

        return dice

    def dice_loss_single_channel(self, input, target):
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()

        return 1 - ((2. * intersection + self.smooth) /
                    (input_flat.sum() + target_flat.sum() + self.smooth))


class Concat_BCEWithLogitsLoss(nn.Module):
    def __init__(self, top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Concat_BCEWithLogitsLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, dic_tmp, y, step):
        total_loss = []
        for i in range(len(dic_tmp)):
            pred_logits = dic_tmp[i]
            gts = y[i]
            if self.top_k_percent_pixels == None:
                final_loss = self.bceloss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2) * pred_logits.size(3))
                pred_logits = pred_logits.view(-1, pred_logits.size(
                    1), pred_logits.size(2) * pred_logits.size(3))
                gts = gts.view(-1, gts.size(1), gts.size(2) * gts.size(3))
                pixel_losses = self.bceloss(pred_logits, gts.float())
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(
                        1.0, step / float(self.hard_example_mining_step))
                    top_k_pixels = int(
                        (ratio * self.top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
                _, top_k_indices = torch.topk(
                    pixel_losses, k=top_k_pixels, dim=2)

                final_loss = nn.BCEWithLogitsLoss(
                    weight=top_k_indices, reduction='mean')(pred_logits, gts)
            final_loss = final_loss.unsqueeze(0)
            total_loss.append(final_loss)
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss


class Concat_CrossEntropyLoss(nn.Module):
    def __init__(self, top_k_percent_pixels=None,
                 hard_example_mining_step=100000):
        super(Concat_CrossEntropyLoss, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        if top_k_percent_pixels is not None:
            assert(top_k_percent_pixels > 0 and top_k_percent_pixels < 1)
        self.hard_example_mining_step = hard_example_mining_step
        if self.top_k_percent_pixels == None:
            self.celoss = nn.CrossEntropyLoss(
                ignore_index=255, reduction='mean')
        else:
            self.celoss = nn.CrossEntropyLoss(
                ignore_index=255, reduction='none')

    def forward(self, dic_tmp, y, step):
        total_loss = []
        for i in range(len(dic_tmp)):
            pred_logits = dic_tmp[i]
            gts = y[i]
            if self.top_k_percent_pixels == None:
                final_loss = self.celoss(pred_logits, gts)
            else:
                # Only compute the loss for top k percent pixels.
                # First, compute the loss for all pixels. Note we do not put the loss
                # to loss_collection and set reduction = None to keep the shape.
                num_pixels = float(pred_logits.size(2) * pred_logits.size(3))
                pred_logits = pred_logits.view(-1, pred_logits.size(
                    1), pred_logits.size(2) * pred_logits.size(3))
                gts = gts.view(-1, gts.size(1) * gts.size(2))
                pixel_losses = self.celoss(pred_logits, gts)
                if self.hard_example_mining_step == 0:
                    top_k_pixels = int(self.top_k_percent_pixels * num_pixels)
                else:
                    ratio = min(
                        1.0, step / float(self.hard_example_mining_step))
                    top_k_pixels = int(
                        (ratio * self.top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
                top_k_loss, top_k_indices = torch.topk(
                    pixel_losses, k=top_k_pixels, dim=1)

                final_loss = torch.mean(top_k_loss)
            final_loss = final_loss.unsqueeze(0)
            total_loss.append(final_loss)
        total_loss = torch.cat(total_loss, dim=0)
        return total_loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True, softmax=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.softmax = True

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        # print(num_class)
        if self.softmax:
            logit = torch.softmax(logit, dim=1)
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')
        
        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss



class OhemCELoss(nn.Module):

    def __init__(self, thresh=0.7, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

class OhemBCELoss(nn.Module):

    def __init__(self, thresh=0.7):
        super(OhemBCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.criteria = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16
        n_min = int(labels.numel() * 0.15)
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


class OhemBCE_Dice(nn.Module):
    def __init__(self,thresh=0.6, alpha=0.9) -> None:
        super().__init__()
        self.bce = OhemBCELoss(thresh)
        self.dice = DiceLoss()
        self.alpha = alpha
    
    def forward(self,logits, labels):
        bce_loss = self.bce(logits, labels)
        dice_loss = self.dice(logits, labels)
        loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss
        return loss


class LovaszSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,logits, labels):
        out = torch.sigmoid(logits)
        loss = L.lovasz_softmax(out, labels.squeeze(1), classes=[1], ignore=255)
        return loss

class LovaszSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, labels):
        out = torch.softmax(logits, dim=1)
        loss = L.lovasz_softmax(out, labels, ignore=255)
        return loss

    

class DistillationLoss(nn.Module):
    def __init__(self, T=1):
        super().__init__()
        self.kd_loss = nn.KLDivLoss()
        # self.l2_loss = nn.MSELoss()
        self.T = T
        
    def forward(self,student_logit, teacher_logit):
        kd_loss = self.kd_loss(F.log_softmax(student_logit/ self.T, dim=1), torch.softmax(teacher_logit / self.T, dim=1))
        # l2_loss = self.l2_loss(student_logit, teacher_logit)

        return kd_loss
