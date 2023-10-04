# !pip install timm
from importlib.util import LazyLoader
from math import ceil, sqrt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import os
from einops import rearrange
from misc import NestedTensor, is_main_process


cfg_small_k400 = {
    'UNIFORMER':{
    "EMBED_DIM": [64, 128, 320, 512],
    "DEPTH": [3, 4, 8, 3],
    "QKV_BIAS": True,
    "QKV_SCALE": None,
    "REPRESENTATION_SIZE": None,
    "HEAD_DIM": 64,
    "MLP_RATIO": 4,
    "DROPOUT_RATE": 0,
    "ATTENTION_DROPOUT_RATE": 0,
    "DROP_DEPTH_RATE": 0.1,
    "SPLIT": False,
    "STD": False,
    "PRETRAIN_NAME": 'uniformer_small_in1k',
    "REMOVE_LAST": True,
    },
    'MODEL':{
        "USE_CHECKPOINT": False,
        "CHECKPOINT_NUM": [0, 0, 0, 0],
        "NUM_CLASSES": 0,
    },
    'DATA':{
        "TRAIN_CROP_SIZE": 224,
        "INPUT_CHANNEL_NUM":[3],
    }
}

model_path = 'path_to_models'
model_path = {
    'uniformer_small_in1k': os.path.join(model_path, 'uniformer_small_in1k.pth'),
    'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    'uniformer_small_k600_16x4': os.path.join(model_path, 'uniformer_small_k600_16x4.pth'),
    'uniformer_base_in1k': os.path.join(model_path, 'uniformer_base_in1k.pth'),
    'uniformer_base_k400_8x8': os.path.join(model_path, 'uniformer_base_k400_8x8.pth'),
    'uniformer_base_k400_16x4': os.path.join(model_path, 'uniformer_base_k400_16x4.pth'),
    'uniformer_base_k600_16x4': os.path.join(model_path, 'uniformer_base_k600_16x4.pth'),
}


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)
    
def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)

def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)

def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)

def bn_3d(dim):
    return nn.BatchNorm3d(dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x   


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x    


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        # print(patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        #FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)

        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x
    

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class Uniformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, cfg):
        super().__init__()

        depth = cfg["UNIFORMER"]["DEPTH"]
        embed_dim = cfg["UNIFORMER"]["EMBED_DIM"]
        head_dim = cfg["UNIFORMER"]["HEAD_DIM"]
        mlp_ratio = cfg["UNIFORMER"]["MLP_RATIO"]
        qkv_bias = cfg["UNIFORMER"]["QKV_BIAS"]
        qk_scale = cfg["UNIFORMER"]["QKV_SCALE"]
        representation_size = cfg["UNIFORMER"]["REPRESENTATION_SIZE"]
        drop_rate = cfg["UNIFORMER"]["DROPOUT_RATE"]
        attn_drop_rate = cfg["UNIFORMER"]["ATTENTION_DROPOUT_RATE"]
        drop_path_rate = cfg["UNIFORMER"]["DROP_DEPTH_RATE"]
        split = cfg["UNIFORMER"]["SPLIT"]
        std = cfg["UNIFORMER"]["STD"]
        self.use_checkpoint = cfg["MODEL"]["USE_CHECKPOINT"]
        self.checkpoint_num = cfg["MODEL"]["CHECKPOINT_NUM"]
        num_classes = cfg["MODEL"]["NUM_CLASSES"] 
        img_size = cfg["DATA"]["TRAIN_CROP_SIZE"]
        in_chans = cfg["DATA"]["INPUT_CHANNEL_NUM"][0]
        remove_last_stage = cfg['UNIFORMER']["REMOVE_LAST"]
        self.stage_num = len(depth) - 1 if remove_last_stage else len(depth)
        # logger.info(f'Use checkpoint: {self.use_checkpoint}')
        # logger.info(f'Checkpoint number: {self.checkpoint_num}')

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6) 
        
        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        if remove_last_stage:
            self.patch_embed4 = None
        else:
            self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        if split:
            self.blocks3 = nn.ModuleList([
                SplitSABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])
            if remove_last_stage:
                self.blocks4 = nn.Identity()
            else:
                self.blocks4 = nn.ModuleList([
                    SplitSABlock(
                        dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
                for i in range(depth[3])])
        else:
            self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
                for i in range(depth[2])])

        
        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def inflate_weight(self, weight_2d, time_dim, center=False):
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def get_pretrained_model(self, cfg):
        if cfg["UNIFORMER"]["PRETRAIN_NAME"]:
            checkpoint = torch.load(model_path[cfg["UNIFORMER"]["PRETRAIN_NAME"]], map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            elif 'model_state' in checkpoint:
                checkpoint = checkpoint['model_state']

            state_dict_3d = self.state_dict()
            for k in checkpoint.keys():
                if checkpoint[k].shape != state_dict_3d[k].shape:
                    # if len(state_dict_3d[k].shape) <= 2:
                    #     logger.info(f'Ignore: {k}')
                    #     continue
                    # logger.info(f'Inflate: {k}, {checkpoint[k].shape} => {state_dict_3d[k].shape}')
                    time_dim = state_dict_3d[k].shape[2]
                    checkpoint[k] = self.inflate_weight(checkpoint[k], time_dim)

            if self.num_classes != checkpoint['head.weight'].shape[0]:
                del checkpoint['head.weight'] 
                del checkpoint['head.bias'] 
            return checkpoint
        else:
            return None

    def smart_load(self, pretrained_dir):
        pretrained_dict = torch.load(pretrained_dir)

        patch_embed_weight = pretrained_dict['patch_embed1.proj.weight']
        patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
        pretrained_dict['patch_embed.proj.weight'] = patch_embed_weight
        # print(pretrained_dict)
        cur_dict = self.state_dict()
        loaded_keys = []
        removed_keys = []

        for k, v in pretrained_dict.items():
            if k in cur_dict.keys():
                if v.shape == cur_dict[k].shape:
                    cur_dict[k] = v
                    loaded_keys.append(k)
                else:
                    removed_keys.append(k)
            else:
                removed_keys.append(k)
        self.load_state_dict(cur_dict)
        print("Missing keys: ",removed_keys)

    def forward_features(self, x, out_all=True):
        # print(x.shape)
        x = self.patch_embed1(x)
        # print(x.shape)
        x = self.pos_drop(x)
        if out_all: outs = []
        for i, blk in enumerate(self.blocks1):
            if self.use_checkpoint and i < self.checkpoint_num[0]:
                x = checkpoint.checkpoint(blk, x)
            else:
                # print(x.shape)
                x = blk(x)
        # print(x.shape) # torch.Size([2, 64, 2, 90, 120]) # 1/4 downsample
        if out_all: outs.append(x)
        x = self.patch_embed2(x)
        for i, blk in enumerate(self.blocks2):
            if self.use_checkpoint and i < self.checkpoint_num[1]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # print(x.shape) # torch.Size([2, 128, 2, 45, 60]) # 1/8 downsample
        if out_all: outs.append(x)

        x = self.patch_embed3(x)
        for i, blk in enumerate(self.blocks3):
            if self.use_checkpoint and i < self.checkpoint_num[2]:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        # print(x.shape) # torch.Size([2, 320, 2, 22, 30]) # 1/16 downsample
        if out_all: outs.append(x)


        return tuple(outs)

    def forward(self, samples: NestedTensor):
        
        vid_frames = rearrange(samples.tensors, 't b c h w -> b c t h w')
        # print(vid_frames.shape)
        vid_embeds = self.forward_features(vid_frames)
        layer_outputs = []
        for o in vid_embeds:
            # print(o.shape)
            tmp = rearrange(o, 'b c t h w -> t b c h w')
            layer_outputs.append(tmp)
        outputs = [] 
        orig_pad_mask = samples.mask
        for l_out in layer_outputs:
            pad_mask = F.interpolate(orig_pad_mask.float(), size=l_out.shape[-2:]).to(torch.bool)
            outputs.append(NestedTensor(l_out, pad_mask))
        return outputs

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def uniformer_small_in1k():
    pass


def uniformer_small_k400(pretrained=None, num_classes=400, remove_last_stage=False):
    # 'uniformer_small_k400_8x8': os.path.join(model_path, 'uniformer_small_k400_8x8.pth'),
    # 'uniformer_small_k400_16x4': os.path.join(model_path, 'uniformer_small_k400_16x4.pth'),
    cfg = {
        'UNIFORMER':{
        "EMBED_DIM": [64, 128, 320, 512],
        "DEPTH": [3, 4, 8, 3],
        "QKV_BIAS": True,
        "QKV_SCALE": None,
        "REPRESENTATION_SIZE": None,
        "HEAD_DIM": 64,
        "MLP_RATIO": 4,
        "DROPOUT_RATE": 0,
        "ATTENTION_DROPOUT_RATE": 0,
        "DROP_DEPTH_RATE": 0.1,
        "SPLIT": False,
        "STD": False,
        "PRETRAIN_NAME": 'uniformer_small_in1k',
        "REMOVE_LAST": remove_last_stage,
        },
        'MODEL':{
            "USE_CHECKPOINT": False,
            "CHECKPOINT_NUM": [0, 0, 0, 0],
            "NUM_CLASSES": num_classes,
        },
        'DATA':{
            "TRAIN_CROP_SIZE": 224,
            "INPUT_CHANNEL_NUM":[3],
        }
    }
    model = Uniformer(cfg)
    if pretrained:
        model.smart_load(pretrained_dir=pretrained)
        print("loading pretrained model parameters from {}".format(pretrained))
    return model

def uniformer_small_k600(pretrained=None, num_classes=600, remove_last_stage=False):
    cfg = {
        'UNIFORMER':{
        "EMBED_DIM": [64, 128, 320, 512],
        "DEPTH": [3, 4, 8, 3],
        "QKV_BIAS": True,
        "QKV_SCALE": None,
        "REPRESENTATION_SIZE": None,
        "HEAD_DIM": 64,
        "MLP_RATIO": 4,
        "DROPOUT_RATE": 0,
        "ATTENTION_DROPOUT_RATE": 0,
        "DROP_DEPTH_RATE": 0.1,
        "SPLIT": False,
        "STD": False,
        "PRETRAIN_NAME": 'uniformer_small_in1k',
        "REMOVE_LAST": remove_last_stage,
        },
        'MODEL':{
            "USE_CHECKPOINT": False,
            "CHECKPOINT_NUM": [0, 0, 0, 0],
            "NUM_CLASSES": num_classes,
        },
        'DATA':{
            "TRAIN_CROP_SIZE": 224,
            "INPUT_CHANNEL_NUM":[3],
        }
    }
    model = Uniformer(cfg)
    if pretrained:
        model.smart_load(pretrained_dir=pretrained)
        print("loading pretrained model parameters from {}".format(pretrained))
    return model

def uniformer_base_k400(pretrained=None, num_classes=400, remove_last_stage=False):
    # 'uniformer_base_k600_16x4'
    cfg = {
        'UNIFORMER':{
            "EMBED_DIM": [64, 128, 320, 512],
            "DEPTH": [5, 8, 20, 7],
            "QKV_BIAS": True,
            "QKV_SCALE": None,
            "REPRESENTATION_SIZE": None,
            "HEAD_DIM": 64,
            "MLP_RATIO": 4,
            "DROPOUT_RATE": 0,
            "ATTENTION_DROPOUT_RATE": 0,
            "DROP_DEPTH_RATE": 0.1,
            "SPLIT": False,
            "STD": False,
            "PRETRAIN_NAME": 'uniformer_small_in1k',
            "REMOVE_LAST": remove_last_stage,
        },
        'MODEL':{
            "USE_CHECKPOINT": False,
            "CHECKPOINT_NUM": [0, 0, 0, 0],
            "NUM_CLASSES": num_classes,
        },
        'DATA':{
            "TRAIN_CROP_SIZE": 224,
            "INPUT_CHANNEL_NUM":[3],
        }
    }
    model = Uniformer(cfg)
    if pretrained:
        model.smart_load(pretrained_dir=pretrained)
        print("loading pretrained model parameters from {}".format(pretrained))
    return model


class VideoUniformerBackbone(nn.Module):
    """
    A wrapper which allows using Video-Swin Transformer as a temporal encoder for MTTR.
    Check out video-swin's original paper at: https://arxiv.org/abs/2106.13230 for more info about this architecture.
    Only the 'tiny' version of video swin was tested and is currently supported in our project.
    Additionally, we slightly modify video-swin to make it output per-frame embeddings as required by MTTR (check our
    paper's supplementary for more details), and completely discard of its 4th block.
    """
    def __init__(self, backbone_pretrained, backbone_pretrained_path, train_backbone, running_mode, **kwargs):
        super(VideoUniformerBackbone, self).__init__()
        # patch_size is (1, 4, 4) instead of the original (2, 4, 4).
        # this prevents swinT's original temporal downsampling so we can get per-frame features.
        self.uniformer_backbone = Uniformer(cfg_small_k400)
        self.uniformer_backbone.patch_embed1 =  nn.Conv3d(3, 64, (1, 4, 4), (1, 4, 4), (0,0,0))
        if backbone_pretrained and running_mode =='train':
            self.uniformer_backbone.smart_load(backbone_pretrained_path)

        self.layer_output_channels = [64, 128, 320]
        self.train_backbone = train_backbone
        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, samples: NestedTensor):
        outputs = self.uniformer_backbone(samples)
        return outputs

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)