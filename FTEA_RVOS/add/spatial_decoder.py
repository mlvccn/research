# import imp
# from turtle import forward
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model
from torch import Tensor
from typing import List
from einops import rearrange
import math

class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        # b, t, c, h, w
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)  # b, t, c, h, w
        return pos

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class FeatureSelectionModule(nn.Module):
    # https://github.com/EMI-Group/FaPN
    # https://arxiv.org/pdf/2108.07058.pdf
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)

        
        # weight_init.c2_xavier_fill(self.conv_atten)
        # weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FFN(nn.Module):
    def __init__(self, c_in, c_hid, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(c_in, c_hid)
        self.act = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(c_hid, c_in)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): input tensor, shape=[b, c, h, w]

        Returns:
            tensor: shape=[b, c, h, w]
        """
        h,w = x.shape[-2:]
        f = rearrange(x, 'b c h w-> b (h w) c')
        y = self.linear2(self.dropout1(self.act(self.linear1(f))))
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
        y = x + self.dropout2(y)
        y = self.norm(y)
        return y

class ConvFFN(nn.Module):
    def __init__(self, c_in, c_hid, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_hid, 1)
        self.act = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_hid, c_in, 1)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        y = self.conv2(self.dropout1(self.act(self.conv1(x))))
        y = x + self.dropout2(y)
        y = self.norm(y)
        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PGConvFFN(nn.Module):
    """_summary_

    Args:
        nn (_type_): point-wise convolution(c->4c) -> group-wise convolution(4c->c)
    """
    def __init__(self, c_in, c_hid, groups, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_hid, 1)
        self.act = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_hid, c_in, 3, padding=1, groups=groups)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        y = self.conv2(self.dropout1(self.act(self.conv1(x))))
        y = x + self.dropout2(y)
        y = self.norm(y)
        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GConvFFN(nn.Module):
    """_summary_

    Args:
        nn (_type_): group-wise convolution(c->4c) -> group-wise convolution(4c->c)
    """
    def __init__(self, c_in, c_hid, groups, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_hid, 3, padding=1, groups=groups)
        self.act = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_hid, c_in, 3, padding=1, groups=groups)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        y = self.conv2(self.dropout1(self.act(self.conv1(x))))
        y = x + self.dropout2(y)
        y = self.norm(y)
        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FPNSpatialDecoderV3(nn.Module):
    """
    An FPN-like spatial decoder. Generates high-res, semantically rich features which serve as the base for creating
    instance segmentation masks.
    """
    def __init__(self, context_dim, fpn_dims, mask_kernels_dim=8):
        super().__init__()

        inter_dims = [context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        # print(inter_dims, fpn_dims)
        # [256, 128, 64, 32, 16] [192, 96]
        self.lay1 = nn.Sequential(
            LayerNorm(context_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(context_dim, inter_dims[0], kernel_size=1, stride=1),
            Block(inter_dims[0])
        )
        self.lay2 = nn.Sequential(
            LayerNorm(inter_dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inter_dims[0], inter_dims[1], kernel_size=1, stride=1),
            Block(inter_dims[1])
        )

        self.lay3 = nn.Sequential(
            LayerNorm(inter_dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inter_dims[1], inter_dims[2], kernel_size=1, stride=1),
            Block(inter_dims[2])
        )
        self.lay4 = nn.Sequential(
            LayerNorm(inter_dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inter_dims[2], inter_dims[3], kernel_size=1, stride=1),
            Block(inter_dims[3])
        )
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.context_dim = context_dim

        self.add_extra_layer = len(fpn_dims) == 3
        if self.add_extra_layer:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
            self.lay5 = nn.Sequential(
                LayerNorm(inter_dims[3], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(inter_dims[3], inter_dims[4], kernel_size=1, stride=1),
                Block(inter_dims[4])
             )
            self.out_lay = torch.nn.Conv2d(inter_dims[4], mask_kernels_dim, 3, padding=1)
        else:
            self.out_lay = torch.nn.Conv2d(inter_dims[3], mask_kernels_dim, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x: Tensor, layer_features: List[Tensor]):
    def forward(self, x: Tensor, layer_features: List[Tensor]):
        x = self.lay1(x)
        x = self.lay2(x)
        cur_fpn = self.adapter1(layer_features[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)

        cur_fpn = self.adapter2(layer_features[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)

        if self.add_extra_layer:
            cur_fpn = self.adapter3(layer_features[2])
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            x = self.lay5(x)

        x = self.out_lay(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim*4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

def attention(q, k, v):
    """[summary]

    Args:
        q ([tensor]): [shape is b, c, h, w]
        k ([tensor]): [shape is b, c, h, w]
        v ([tensor]): [shape is b, c, h, w]
    """
    c, h,w = q.shape[-3:]
    out = None
    q = rearrange(q, 'b c h w-> b (h w) c')
    k = rearrange(k, 'b c h w-> b c (h w)')
    v = rearrange(v, 'b c h w-> b (h w) c')
    attention_map = torch.bmm(q, k) / math.sqrt(c)
    attention_map = torch.softmax(attention_map, dim=-1) 
    z = torch.bmm(attention_map, v)
    out = rearrange(z, 'b (h w) c-> b c h w', h=h, w=w)
    return out

class ObjectFilter(nn.Module):
    """a block of decoder
    using transformer style to do both upsampling and filtering objects
    """
    def __init__(self, c_skip, c_low, c_out, drop_path=0.,  object_candidates=50,  mask_kernels_dim=8) -> None:
        super().__init__()
        self.norm = LayerNorm(c_skip, eps=1e-6, data_format="channels_first")
        self.conv_q = nn.Sequential(
            nn.Conv2d(c_skip, mask_kernels_dim, 1),
            nn.GELU()
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(c_low, mask_kernels_dim, 1),
            nn.GELU()
        )
        self.conv_v = nn.Conv2d(c_low, c_out, 1)

        self.conv_post = nn.Conv2d(c_out, c_out, 1)

        self.mlp = nn.Sequential(
            LayerNorm(c_out, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(c_out, c_out*4, 1),
            nn.GELU(),
            nn.Conv2d(c_out*4, c_out, 1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_low, f_skip, kernel):
        """[summary]

        Args:
            x_low ([tensor]): [shape is b c h w]
            f_skip ([tensor]): [shape is b c 2h 2w]
            kernel ([tensor]): [shape is b n c]

        Returns:
            [tensor]: [shape is b c 2h 2w]
        """
        h,w = f_skip.shape[-2:]
        q = self.conv_q(self.norm(f_skip))
#        print(q.shape, kernel.shape)
        q = torch.einsum("bchw, bnc->bnhw",q, kernel)

        k = self.conv_k(x_low)
        k = torch.einsum("bchw, bnc->bnhw", k, kernel)
        
        v = self.conv_v(x_low)

        v_up = F.interpolate(v, size=(h,w), mode='nearest')
        x1 = attention(q, k, v)
        x1 = self.conv_post(x1)
        x1 = x1 + v_up

        x2 = self.mlp(x1)
        return x1 + self.drop_path(x2)


class ObjectGroupFilter(nn.Module):
    """a block of decoder
    using transformer style to do both upsampling and filtering objects
    """
    def __init__(self, c_skip, c_low, c_out, drop_path=0., object_candidates=50, mask_kernels_dim=8) -> None:
        super().__init__()
        self.group = int(c_out/mask_kernels_dim)
        self.mask_kernels_dim = mask_kernels_dim
        
        self.norm1 = LayerNorm(c_skip, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(c_low, eps=1e-6, data_format="channels_first")

        self.conv_q = nn.Sequential(
            nn.Conv2d(c_skip, c_out, 1),
            nn.GELU()
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(c_low, c_out, 1),
            nn.GELU()
        )
        self.conv_v = nn.Conv2d(c_low, object_candidates * self.group, 1)

        self.conv_post = nn.Conv2d(object_candidates * self.group, object_candidates * self.group, 1)
        d_out = object_candidates * self.group
        self.mlp = nn.Sequential(
            LayerNorm(d_out, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(d_out, d_out*4, 1),
            nn.GELU(),
            nn.Conv2d(d_out*4, d_out, 1)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x_low, f_skip, kernel):
        """[summary]

        Args:
            x_low ([tensor]): [shape is b c h w]
            f_skip ([tensor]): [shape is b c 2h 2w]
            kernel ([tensor]): [shape is b n c]

        Returns:
            [tensor]: [shape is b c 2h 2w]
        """
        b,_,h1,w1 = f_skip.shape
        h2,w2 = x_low.shape[-2:]
        x_low = self.norm2(x_low)
        f_skip = self.norm1(f_skip)
        
        q = self.conv_q(f_skip)
        q = q.view(b, self.group, self.mask_kernels_dim, h1, w1)
        q = torch.einsum("bgchw, bnc->bgnhw",q, kernel)
        q = rearrange(q, 'b g n h w-> (b g) n h w')
        
        k = self.conv_k(x_low)
        k = k.view(b, self.group, self.mask_kernels_dim, h2, w2)
        k = torch.einsum("bgchw, bnc->bgnhw", k, kernel)
        k = rearrange(k, 'b g n h w-> (b g) n h w')

        v_ = self.conv_v(x_low)
        v = rearrange(v_, 'b (g n) h w-> (b g) n h w',b=b, g=self.group)

        v_up = F.interpolate(v_, size=(h1,w1), mode='bilinear', align_corners=False)
        # print(q.shape, k.shape, v.shape)
        x1 = attention(q, k, v)
        # print(x1.shape)
        x1 = rearrange(x1, '(b g) n h w->b (g n) h w', b=b, g=self.group)
        x1 = self.conv_post(x1)
        # print(x1.shape, v_up.shape)
        x1 = x1 + v_up
        x2 = self.mlp(x1)
        return x1 + self.drop_path(x2)


class DecFormer(nn.Module):
    def __init__(self, c_in_low, c_in_skip, obj_num=50, kernel_dim=8, c_ratio=4, dropout=0.1):
        super().__init__()
        c_out = c_ratio * obj_num
        self.conv_q = nn.Conv2d(c_in_skip, obj_num, 1)
        self.conv_k = nn.Conv2d(c_in_low, obj_num, 1)
        self.conv_v = nn.Conv2d(c_in_low, c_ratio * obj_num, 1)

        self.conv_mask = nn.Conv2d(c_in_low, kernel_dim, 1)
        self.norm = LayerNorm(c_out, eps=1e-6, data_format="channels_first")
        self.dropout = nn.Dropout(dropout)
        self.conv_mlp = GConvFFN(c_in=c_out, c_hid=c_out*4, groups=obj_num, dropout=dropout)
    
    def kernel2mask(self, f_low, kernel, size=None):
        """
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()
        """
        f_mask = self.conv_mask(f_low)
        # print(f_mask.shape)
        if size!=None:
            f_mask = F.interpolate(f_mask, size=size, mode="bilinear", align_corners=False)
        # print(kernel.shape, f_mask.shape)
        f_mask_ = torch.einsum('bnc,bchw->bnhw',kernel, f_mask)
        # print(f_mask_.shape)
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        # attn_mask = (torch.sigmoid(f_mask_).flatten(2).unsqueeze(1).flatten(0, 1) > 0.5).float()
        attn_mask = torch.sigmoid(f_mask_).flatten(2).unsqueeze(1).flatten(0, 1) 
        # attn_mask = attn_mask.detach()
        # b,n,hw
        return attn_mask
    
    def forward(self, f_low, f_skip, kernel, mask_q=True):
        q = self.conv_q(f_skip)
        k = self.conv_k(f_low)
        v = self.conv_v(f_low)
        v_residual = F.interpolate(v, size=f_skip.shape[-2:], mode="bilinear", align_corners=False)
        b,c1,h1,w1 = q.shape
        _,c2,h2,w2 = v.shape
        q = q.view(b,c1,h1*w1)
        k = k.view(b,c1,h2*w2)
        # print(q.shape,k.shape)
        # print(kernel.shape)
        # mask out non object region:  query or  key
        attn_mask = self.kernel2mask(f_low, kernel, size=f_skip.shape[-2:] if mask_q else None) # b, n, h*w
        if mask_q:
            q = attn_mask * q
        else:
            k = attn_mask * k
        # attention
        # print(q.shape,k.shape)
        attn_map = torch.bmm(q.transpose(2,1), k) / math.sqrt(c1)
        attn_map = torch.softmax(attn_map, dim=-1)
        v = v.view(b,c2,h2*w2).transpose(2, 1)
        out = torch.bmm(attn_map, v)
        out = out.transpose(1, 2).view(b, c2, h1, w1)
        y = self.dropout(out) + v_residual
        y = self.norm(y)
        # FFN
        z = self.conv_mlp(y)

        return z

class FPNSpatialDecoderV4(nn.Module):
    """[summary]
    v4.8
    Args:
        nn ([type]): [description]
    """
    def __init__(self, context_dim, fpn_dims, mask_kernels_dim=8,object_candidates=50):
        super().__init__()
        inter_dims = [context_dim, context_dim // 4, context_dim // 8, context_dim // 16]
        hid_dim1 = int(inter_dims[1] / mask_kernels_dim) * object_candidates
        hid_dim2 = int(inter_dims[2] / mask_kernels_dim) * object_candidates
        # [256, 64, 32, 16] [192, 96]
        self.block1 = ObjectGroupFilter(fpn_dims[0], inter_dims[0], c_out=inter_dims[1], mask_kernels_dim=mask_kernels_dim)
        self.block2 = ObjectGroupFilter(fpn_dims[1], hid_dim1, c_out=inter_dims[2], mask_kernels_dim=mask_kernels_dim)
        if len(fpn_dims) == 3:
            self.extra_layer = True
            self.block3 = ObjectFilter(fpn_dims[2], hid_dim2, c_out=inter_dims[3])
            hid_dim3 = int(inter_dims[3] / mask_kernels_dim) * object_candidates
            self.out_lay = torch.nn.Conv2d(hid_dim3, mask_kernels_dim, 3, padding=1)
        else:
            self.out_lay = torch.nn.Conv2d(hid_dim2, mask_kernels_dim, 3, padding=1)
            self.extra_layer = False

    def forward(self, x: Tensor, layer_features: List[Tensor], kernels: Tensor):
        """[summary]

        Args:
            x (Tensor): [description]
            layer_features (List[Tensor]): [description]
            kernels (Tensor): [shape is l t b n c]

        Returns:
            [type]: [description]
        """
        kernels = rearrange(kernels, 'l t b n c-> l (t b) n c')
        # print(x.shape, kernels[-1,...].shape)
        y1 = self.block1(x, layer_features[0], kernels[0,...])
        
        y2 = self.block2(y1, layer_features[1], kernels[1,...])
        if self.extra_layer:
            y3 = self.block3(y2, layer_features[3], kernels[-1,...])
        else:
            y3 = y2

        y = self.out_lay(y3)
        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DecFormerT1(nn.Module):
    def __init__(self, c_in, obj_num=50, dropout=0.1):
        super().__init__()
        self.conv_q = nn.Linear(c_in, c_in)
        self.conv_k = nn.Linear(c_in, c_in)
        self.conv_v = nn.Linear(c_in, c_in)

        self.norm = LayerNorm(c_in, eps=1e-6, data_format="channels_first")
        self.dropout = nn.Dropout(dropout)
        self.ffn = GConvFFN(c_in=c_in, c_hid=c_in*4, groups=obj_num, dropout=dropout)
        N_steps = c_in // 2
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)

    def forward(self, x):
        """_summary_

        Args:
            x (temsor): shape=[b,t,c,h,w]

        Returns:
            _type_: _description_
        """
        b,t,c,h,w = x.shape
        x_residual = x
        pos = self.pe_layer(x)
        pos = rearrange(pos,'b t c h w-> b (t h w) c' )
        # print(pos.shape, x.shape)
        x = rearrange(x, 'b t c h w-> b (t h w) c')
        q = self.conv_q(x) + pos
        k = self.conv_k(x) + pos
        v = self.conv_v(x)
        c1 = q.shape[-1]
        # attention
        attn_map = torch.bmm(q, k.transpose(2,1)) / math.sqrt(c1) # bx(thw)x(thw)
        attn_map = torch.softmax(attn_map, dim=-1)
        out = torch.bmm(attn_map, v)
        out = rearrange(out, 'b (t h w) c->b t c h w', t=t, h=h, w=w)
        y = self.dropout(out) + x_residual
        y = rearrange(y, 'b t c h w-> (b t) c h w')
        y = self.norm(y)
        # FFN
        z = self.ffn(y)
        z = rearrange(z, '(b t) c h w-> b t c h w', b=b, t=t)
        return z


class FPNSpatialDecoderV5(nn.Module):
    """[summary]
    v5.0: channel 200->100
    Args:
        nn ([type]): [description]
    #params=194,472
    """
    def __init__(self, context_dim, fpn_dims, 
        mask_kernels_dim=8,object_candidates=50, c_ratio=4, dropout=0.1):
        super().__init__()

        self.block1 = DecFormer(
            c_in_low=context_dim, 
            c_in_skip=fpn_dims[0],
            obj_num=object_candidates,
            kernel_dim=mask_kernels_dim,
            c_ratio=c_ratio,
            dropout=dropout
        )
        self.block2 = DecFormer(
            c_in_low=object_candidates * c_ratio, 
            c_in_skip=fpn_dims[1],
            obj_num=object_candidates,
            kernel_dim=mask_kernels_dim,
            c_ratio=c_ratio//2,
            dropout=dropout
        )
        self.out_lay = nn.Conv2d(object_candidates* c_ratio//2, mask_kernels_dim, 3, padding=1)

    def forward(self, x: Tensor, layer_features: List[Tensor], kernels: Tensor):
    # def forward(self, input_):
        """[summary]

        Args:
            x (Tensor): [description]
            layer_features (List[Tensor]): [description]
            kernels (Tensor): [shape is l t b n c]

        Returns:
            [type]: [description]
        """
        # x = input_['x']
        # layer_features = input_['layer_features']
        # kernels = input_['kernels']
        kernels = rearrange(kernels, 'l t b n c-> l (t b) n c')
        # print(kernels.shape)
        # print(x.shape,layer_features[0].shape, kernels[-1,...].shape)
        # torch.Size([3, 256, 20, 27]) torch.Size([3, 192, 40, 54]) torch.Size([3, 50, 8])
        # torch.Size([3, 200, 40, 54]) torch.Size([3, 96, 80, 107]) torch.Size([3, 50, 8])
    
        y1 = self.block1(x, layer_features[0], kernels[0,...])
        # print(y1.shape,layer_features[1].shape, kernels[-1,...].shape)
        y2 = self.block2(y1, layer_features[1], kernels[1,...])
        y = self.out_lay(y2)
        return y

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DynamicHead(nn.Module):
    def __init__(self, kernel_size=1, obj_num=50) -> None:
        super().__init__()
        self.obj_num = obj_num
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def generate_coor_feat(self,batch, size, device):
        """_summary_

        Args:
            kernel (tensor): object kernel, shape=[b, n, e]
            size (int, int): mask feature resolution, height and width,

        Returns:
            tensor: coordinates feature, shape = [b, 2, h, w]
        """
        h, w = size
        x_range = torch.linspace(-1, 1, w, device=device)
        y_range = torch.linspace(-1, 1, h, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([batch, 1, -1, -1])
        x = x.expand([batch, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        return coord_feat

    def forward(self, f, kernel):
        """_summary_

        Args:
            f (tensor): mask feature map, shape = [b, c, h, w]
            kernel (tensor): object convolution kernel, shape = [b, n, e]

        Returns:
            tensor: predicted masks, shape=[b, n, h, w]
            n is the object number, each object mask is a binary matrix
        """
        mask_pred = None 
        batch, _, h, w = f.shape

        # mask feature change: [b, c, h, w]
        # c -> n x r: n objects, r channel feature for each objects

        # kernel change: [b, n, e] --> [b, n, r, k, k]
        # e is object feature dimension, converting into 1x1 convolution kernel with 
        # input channel r and output channel 1
        coord_feat = self.generate_coor_feat(batch=batch, size=(h, w), device=kernel.device)

        #  group-wise convolution based mask prediction
        outs = []
        for i in range(batch):
            tmp = F.conv2d(f[i:i+1, ...], kernel[i], padding=self.padding, groups=self.obj_num)
            outs.append(tmp)
        mask_pred = torch.cat(outs, dim=0)

        return mask_pred



class FPNSpatialDecoderV3_2(nn.Module):
    """
    An FPN-like spatial decoder. Generates high-res, semantically rich features which serve as the base for creating
    instance segmentation masks.
    """
    def __init__(self, context_dim, fpn_dims, mask_kernels_dim=8, object_candidates=50):
        super().__init__()

        inter_dims = [context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        # print(inter_dims, fpn_dims)
        # [256, 128, 64, 32, 16] [192, 96]
        self.lay1 = nn.Sequential(
            LayerNorm(context_dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(context_dim, inter_dims[0], kernel_size=1, stride=1),
            Block(inter_dims[0])
        )
        self.lay2 = nn.Sequential(
            LayerNorm(inter_dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(inter_dims[0], inter_dims[1], kernel_size=1, stride=1),
            Block(inter_dims[1])
        )

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.linear3 = nn.Linear(mask_kernels_dim, inter_dims[1])
        self.norm3 = LayerNorm(object_candidates, eps=1e-6, data_format="channels_first")
        self.blk3 = Block(object_candidates)

        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], object_candidates, 1)
        self.linear4 = nn.Linear(mask_kernels_dim, object_candidates)
        self.norm4 = LayerNorm(object_candidates, eps=1e-6, data_format="channels_first")
        self.blk4 = Block(object_candidates)

        self.context_dim = context_dim

        self.add_extra_layer = len(fpn_dims) == 3
        if self.add_extra_layer:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
            self.lay5 = nn.Sequential(
                LayerNorm(inter_dims[3], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(inter_dims[3], inter_dims[4], kernel_size=1, stride=1),
                Block(inter_dims[4])
             )
            self.out_lay = torch.nn.Conv2d(inter_dims[4], mask_kernels_dim, 3, padding=1)
        else:
            self.out_lay = torch.nn.Conv2d(object_candidates, mask_kernels_dim, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x: Tensor, layer_features: List[Tensor]):
    def forward(self, x: Tensor, layer_features: List[Tensor], kernels: Tensor):
        kernels = rearrange(kernels, 'l t b n c-> l (t b) n c')
        k1 = kernels[0,...] # (tb) n c
        k2 = kernels[1,...]
        x = self.lay1(x)
        x = self.lay2(x)
        cur_fpn = self.adapter1(layer_features[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)

        cur_fpn = self.adapter2(layer_features[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)

        if self.add_extra_layer:
            cur_fpn = self.adapter3(layer_features[2])
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            x = self.lay5(x)

        x = self.out_lay(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# from fvcore.nn import FlopCountAnalysis
# input_ = {
#     'x': torch.rand(1, 256, 20, 27),
#     'layer_features':[
#         torch.rand(1, 192, 40, 54),
#         torch.rand(1, 96, 80, 107)
#     ],
#     'kernels': torch.rand((3,3,1, 50,8 ))
# }
# model = FPNSpatialDecoderV5(context_dim=256, fpn_dims=[192, 96])
# flops = FlopCountAnalysis(model, input_)
# print(flops.total())